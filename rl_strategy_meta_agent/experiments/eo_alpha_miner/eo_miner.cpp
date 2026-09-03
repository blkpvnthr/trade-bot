// EO-backed symbolic alpha miner (research only -- no trading/order code).
//
// Evolutionary backbone: EO's genetic-programming machinery (eoParseTree +
// eoGpDepthInitializer + eoSubtreeXOver + eoBranchMutation + eoSGATransform +
// eoDetTournamentSelect + eoPlusReplacement [elitist] + eoEasyEA). Cribbed from
// the verified-building eo/test/t-eoSymreg.cpp, swapping the RMS target for a
// cross-sectional rank-IC fitness computed over a feature CSV.
//
// The tree nodes are ARITHMETIC only (+, -, *, safe_div, min, max, abs, sign,
// neg) over feature COLUMNS. The time-series operators the spec asks for
// (rolling_mean/std, zscore, rank, delay, delta) are precomputed by the Python
// side as feature columns, so this core stays small and panel-agnostic.
//
// Input CSV: header "group,target,<f0>,<f1>,...", group = int timestamp id,
// target = next-period return, rest = features. Output: top-K unique formulas
// (sexp: add(f0,mul(f1,0.5))) one per line, re-validated by the Python runner.
//
// Build: see build.sh (clang++ -std=c++14, libeo.a libeoutils.a).

#include <gp/eoParseTree.h>
#include <eo>

#include <cmath>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <set>
#include <algorithm>

using namespace gp_parse_tree;
using namespace std;

// ---- dataset (loaded once, global so the fitness functor can see it) ----
static vector<vector<double> > g_rows;   // [row][feature]
static vector<int>             g_group;  // [row] timestamp id (sorted, contiguous blocks)
static vector<double>          g_target; // [row]
static vector<string>          g_names;  // feature names
static int    g_nfeat = 0;
static double g_const_lo = -2.0, g_const_hi = 2.0;
static double g_parsimony = 0.001;       // complexity penalty per node

// ---- alpha expression node ----
class Node {
public:
    enum Op { Var, Const, Add, Sub, Mul, Div, Min2, Max2, Abs, Sign, Neg };

    Node() : op(Var), idx(0), val(0.0) {}
    Node(Op o) : op(o), idx(0), val(0.0) {}

    int arity() const {
        switch (op) {
            case Var: case Const: return 0;
            case Abs: case Sign: case Neg: return 1;
            default: return 2;
        }
    }

    // randomize() is called by the initializer on terminal prototypes.
    void randomize() {
        if (op == Var)        idx = (g_nfeat > 0) ? (int)(rng.random(g_nfeat)) : 0;
        else if (op == Const) val = g_const_lo + rng.uniform() * (g_const_hi - g_const_lo);
    }

    // numeric evaluation: values = pointer to the current feature row
    template <class Children>
    void operator()(double& result, Children args, const vector<double>* row) const {
        double a = 0.0, b = 0.0;
        int n = arity();
        if (n >= 1) args[0].apply(a, row);
        if (n >= 2) args[1].apply(b, row);
        switch (op) {
            case Var:   result = (idx < (int)row->size()) ? (*row)[idx] : 0.0; break;
            case Const: result = val; break;
            case Add:   result = a + b; break;
            case Sub:   result = a - b; break;
            case Mul:   result = a * b; break;
            case Div:   result = (fabs(b) > 1e-9) ? a / b : 0.0; break;   // safe_div
            case Min2:  result = (a < b) ? a : b; break;
            case Max2:  result = (a > b) ? a : b; break;
            case Abs:   result = fabs(a); break;
            case Sign:  result = (a > 0) - (a < 0); break;
            case Neg:   result = -a; break;
        }
        if (!std::isfinite(result)) result = 0.0;                         // NaN-safe
    }

    // formula printing as sexp (parsed back by the Python runner)
    template <class Children>
    void operator()(string& result, Children args) const {
        if (op == Var)   { result = (idx < g_nfeat) ? g_names[idx] : "f0"; return; }
        if (op == Const) { ostringstream o; o << val; result = o.str(); return; }
        string a, b;
        args[0].apply(a);
        if (arity() == 2) args[1].apply(b);
        result = name() + "(" + a + (arity() == 2 ? ("," + b) : "") + ")";
    }

    string name() const {
        switch (op) {
            case Add: return "add"; case Sub: return "sub"; case Mul: return "mul";
            case Div: return "safe_div"; case Min2: return "min"; case Max2: return "max";
            case Abs: return "abs"; case Sign: return "sign"; case Neg: return "neg";
            default: return "x";
        }
    }
    Op getOp() const { return op; }
    int getIdx() const { return idx; }
    double getVal() const { return val; }

private:
    Op op; int idx; double val;
};

// serialization required by eoParseTree (compact: op char [+ payload])
ostream& operator<<(ostream& os, const Node& n) {
    os << (int)n.getOp() << ' ' << n.getIdx() << ' ' << n.getVal();
    return os;
}
istream& operator>>(istream& is, Node& n) {
    int o = 0, i = 0; double v = 0; is >> o >> i >> v;
    n = Node((Node::Op)o);
    return is;
}

typedef eoMaximizingFitness FitnessType;
typedef eoParseTree<FitnessType, Node> Tree;

// per-group (cross-sectional) rank-IC, averaged across timestamps, abs value so
// either polarity is discoverable; minus a parsimony penalty on tree size.
struct ICEval : public eoEvalFunc<Tree> {
    void operator()(Tree& t) {
        size_t N = g_rows.size();
        vector<double> pred(N);
        for (size_t r = 0; r < N; ++r) t.apply(pred[r], &g_rows[r]);

        double sumIC = 0.0; int groups = 0;
        size_t i = 0;
        while (i < N) {
            size_t j = i;
            while (j < N && g_group[j] == g_group[i]) ++j;      // [i,j) = one timestamp
            double ic = rankCorr(pred, g_target, i, j);
            if (std::isfinite(ic)) { sumIC += ic; ++groups; }
            i = j;
        }
        double IC = (groups > 0) ? sumIC / groups : 0.0;
        double fit = fabs(IC) - g_parsimony * (double)t.size();
        if (!std::isfinite(fit)) fit = -1e9;
        t.fitness(fit);
    }

    // Spearman over rows [lo,hi): Pearson on ranks
    static double rankCorr(const vector<double>& x, const vector<double>& y, size_t lo, size_t hi) {
        size_t n = hi - lo;
        if (n < 5) return std::nan("");
        vector<double> rx = ranks(x, lo, hi), ry = ranks(y, lo, hi);
        double mx = 0, my = 0;
        for (size_t k = 0; k < n; ++k) { mx += rx[k]; my += ry[k]; }
        mx /= n; my /= n;
        double sxy = 0, sxx = 0, syy = 0;
        for (size_t k = 0; k < n; ++k) {
            double dx = rx[k] - mx, dy = ry[k] - my;
            sxy += dx * dy; sxx += dx * dx; syy += dy * dy;
        }
        if (sxx < 1e-12 || syy < 1e-12) return std::nan("");   // degenerate (constant)
        return sxy / sqrt(sxx * syy);
    }
    static vector<double> ranks(const vector<double>& v, size_t lo, size_t hi) {
        size_t n = hi - lo;
        vector<size_t> ord(n);
        for (size_t k = 0; k < n; ++k) ord[k] = k;
        sort(ord.begin(), ord.end(), [&](size_t a, size_t b){ return v[lo+a] < v[lo+b]; });
        vector<double> rk(n);
        for (size_t k = 0; k < n; ++k) rk[ord[k]] = (double)k;  // ties broken by order (good enough)
        return rk;
    }
};

// Records the best few unique formulas seen EACH generation, so the final output
// reflects the whole search trajectory (diverse), not just the converged pop.
struct GenArchive : public eoUpdater {
    eoPop<Tree>& P;
    vector<pair<double, string> >& arch;
    GenArchive(eoPop<Tree>& p, vector<pair<double, string> >& a) : P(p), arch(a) {}
    void operator()() {
        vector<size_t> ord(P.size());
        for (size_t i = 0; i < P.size(); ++i) ord[i] = i;
        sort(ord.begin(), ord.end(), [&](size_t a, size_t b){ return P[a].fitness() > P[b].fitness(); });
        int kept = 0;
        for (size_t r = 0; r < ord.size() && kept < 3; ++r) {   // top 3 unique this gen
            string f; P[ord[r]].apply(f);
            bool dup = false;
            for (auto& e : arch) if (e.second == f) { dup = true; break; }
            if (dup) continue;
            arch.push_back(make_pair((double)P[ord[r]].fitness(), f));
            ++kept;
        }
    }
};

static void load_csv(const string& path) {
    ifstream in(path.c_str());
    if (!in) throw runtime_error("cannot open " + path);
    string line; getline(in, line);
    stringstream hs(line); string col;
    vector<string> header;
    while (getline(hs, col, ',')) header.push_back(col);
    for (size_t c = 2; c < header.size(); ++c) g_names.push_back(header[c]);
    g_nfeat = (int)g_names.size();
    while (getline(in, line)) {
        if (line.empty()) continue;
        stringstream ls(line); string cell; vector<double> vals; int c = 0;
        int grp = 0; double tgt = 0;
        while (getline(ls, cell, ',')) {
            double v = cell.empty() ? 0.0 : atof(cell.c_str());
            if (c == 0) grp = (int)v; else if (c == 1) tgt = v; else vals.push_back(v);
            ++c;
        }
        g_group.push_back(grp); g_target.push_back(tgt); g_rows.push_back(vals);
    }
}

int main(int argc, char** argv) {
    try {
        // args: csv out [pop gens maxdepth seed xrate mrate topk]
        if (argc < 3) { cerr << "usage: eo_miner <csv> <out> [pop gens maxdepth seed xrate mrate topk]\n"; return 2; }
        string csv = argv[1], out = argv[2];
        int pop      = argc > 3 ? atoi(argv[3]) : 300;
        int gens     = argc > 4 ? atoi(argv[4]) : 40;
        int maxDepth = argc > 5 ? atoi(argv[5]) : 5;
        int seed     = argc > 6 ? atoi(argv[6]) : 42;
        double xrate = argc > 7 ? atof(argv[7]) : 0.7;
        double mrate = argc > 8 ? atof(argv[8]) : 0.3;
        int topk     = argc > 9 ? atoi(argv[9]) : 25;
        int maxSize  = maxDepth * maxDepth + 8;

        rng.reseed((unsigned)seed);
        load_csv(csv);
        cerr << "loaded " << g_rows.size() << " rows, " << g_nfeat << " features\n";

        // function + terminal set for the GP initializer
        vector<Node> init;
        Node ops[] = { Node(Node::Add), Node(Node::Sub), Node(Node::Mul), Node(Node::Div),
                       Node(Node::Min2), Node(Node::Max2), Node(Node::Abs), Node(Node::Sign),
                       Node(Node::Neg), Node(Node::Var), Node(Node::Const) };
        for (auto& n : ops) init.push_back(n);

        eoGpDepthInitializer<FitnessType, Node> initializer(maxDepth, init);
        ICEval eval;
        eoPop<Tree> P(pop, initializer);
        apply<Tree>(eval, P);

        eoSubtreeXOver<FitnessType, Node>   xover(maxSize);
        eoBranchMutation<FitnessType, Node> mutation(initializer, maxSize);
        eoSGATransform<Tree> transform(xover, xrate, mutation, mrate);

        eoDetTournamentSelect<Tree> selectOne(3);
        eoSelectMany<Tree>          select(selectOne, 1.0);   // fill a full generation
        eoPlusReplacement<Tree>     replace;                  // (mu+lambda) -> elitist
        eoGenContinue<Tree>         term(gens);
        eoCheckPoint<Tree>          checkPoint(term);
        eoBestFitnessStat<Tree>     best;
        checkPoint.add(best);
        vector<pair<double, string> > archive;          // best-per-generation
        GenArchive archiver(P, archive);
        checkPoint.add(archiver);

        eoEasyEA<Tree> gp(checkPoint, eval, select, transform, replace);
        gp(P);

        // emit top-K unique formulas across the whole search, best first
        P.sort();
        for (size_t i = 0; i < P.size(); ++i) {
            string f; P[i].apply(f);
            archive.push_back(make_pair((double)P[i].fitness(), f));
        }
        sort(archive.begin(), archive.end(),
             [](const pair<double,string>& a, const pair<double,string>& b){ return a.first > b.first; });
        ofstream of(out.c_str());
        set<string> seen; int written = 0;
        for (size_t i = 0; i < archive.size() && written < topk; ++i) {
            if (seen.count(archive[i].second)) continue;
            seen.insert(archive[i].second);
            of << archive[i].second << "\n";
            ++written;
        }
        cerr << "wrote " << written << " formulas -> " << out << "\n";
        return 0;
    } catch (exception& e) {
        cerr << "exception: " << e.what() << "\n";
        return 1;
    }
}
