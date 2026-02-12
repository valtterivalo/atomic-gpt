/**
 * @fileoverview gpt.c - The most atomic GPT in pure, dependency-free C.
 *
 * A faithful translation of Karpathy's pure Python GPT with bug fixes
 * and a raw-double inference path (no autograd when we don't need gradients).
 *
 * Compile: gcc -O2 -o gpt gpt.c -lm
 * Run:     ./gpt   (expects input.txt in current directory)
 *
 * Download input.txt:
 *   curl -O https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt
 *   mv names.txt input.txt
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

/* ---- Hyperparameters ---- */
#define N_EMBD     16
#define N_HEAD     4
#define N_LAYER    1
#define BLOCK_SIZE 8
#define HEAD_DIM   (N_EMBD / N_HEAD) /* 4 - exact division, no precision loss */
_Static_assert(N_EMBD % N_HEAD == 0, "N_EMBD must be divisible by N_HEAD");
#define MLP_DIM    (4 * N_EMBD)

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define MAX_DOCS    40000
#define MAX_DOC_LEN 64

/* ---- Allocation wrappers (crash immediately on failure) ---- */

static void *xmalloc(size_t size, const char *context) {
    void *ptr = malloc(size);
    if (ptr == NULL) {
        fprintf(stderr, "allocation failed in %s (size=%zu)\n", context, size);
        exit(1);
    }
    return ptr;
}

static void *xcalloc(size_t count, size_t size, const char *context) {
    void *ptr = calloc(count, size);
    if (ptr == NULL) {
        fprintf(stderr, "allocation failed in %s (count=%zu, size=%zu)\n", context, count, size);
        exit(1);
    }
    return ptr;
}

static void *xrealloc(void *ptr, size_t size, const char *context) {
    void *new_ptr = realloc(ptr, size);
    if (new_ptr == NULL) {
        fprintf(stderr, "allocation failed in %s (size=%zu)\n", context, size);
        exit(1);
    }
    return new_ptr;
}

/* ---- RNG (xorshift64, seeded to 42) ---- */
static uint64_t rng_state = 42;

static double rand_uniform(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (double)(rng_state & 0x1FFFFFFFFFFFFFULL) / (double)0x1FFFFFFFFFFFFFULL;
}

static double rand_gauss(double mean, double std) {
    double u1 = rand_uniform(), u2 = rand_uniform();
    return mean + std * sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

static void shuffle_ints(int *arr, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = (int)(rand_uniform() * (i + 1));
        int tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
    }
}

static int weighted_choice(double *weights, int n) {
    double total = 0;
    for (int i = 0; i < n; i++) total += weights[i];
    double r = rand_uniform() * total, cum = 0;
    for (int i = 0; i < n; i++) { cum += weights[i]; if (r < cum) return i; }
    return n - 1;
}

/* ======================================================================
 * Autograd: Value node and computation graph
 * ====================================================================== */

typedef struct Value {
    double data;
    double grad;
    int n_children;
    struct Value *children[2];
    double local_grads[2];
    int visited; /* epoch counter for topo sort */
} Value;

/* ---- Arena allocator for temporary Values (reset each training step) ---- */
static Value *arena_buf = NULL;
static int arena_count = 0, arena_cap = 0;

static void arena_init(int cap) {
    arena_buf = (Value *)xmalloc(sizeof(Value) * (size_t)cap, "arena_init");
    arena_count = 0;
    arena_cap = cap;
}

static void arena_reset(void) { arena_count = 0; }

static Value *arena_alloc(void) {
    if (arena_count >= arena_cap) {
        arena_cap *= 2;
        arena_buf = (Value *)xrealloc(arena_buf, sizeof(Value) * (size_t)arena_cap, "arena_alloc");
    }
    Value *v = &arena_buf[arena_count++];
    v->data = 0; v->grad = 0; v->n_children = 0; v->visited = 0;
    return v;
}

/* ---- Value constructors ---- */

static Value *val_new(double data) {
    Value *v = arena_alloc();
    v->data = data;
    return v;
}

static Value *param_new(double data) {
    Value *v = (Value *)xcalloc(1, sizeof(Value), "param_new");
    v->data = data;
    return v;
}

/* ---- Value operations ---- */

static Value *val_add(Value *a, Value *b) {
    Value *v = arena_alloc();
    v->data = a->data + b->data;
    v->n_children = 2;
    v->children[0] = a; v->children[1] = b;
    v->local_grads[0] = 1.0; v->local_grads[1] = 1.0;
    return v;
}

static Value *val_mul(Value *a, Value *b) {
    Value *v = arena_alloc();
    v->data = a->data * b->data;
    v->n_children = 2;
    v->children[0] = a; v->children[1] = b;
    v->local_grads[0] = b->data; v->local_grads[1] = a->data;
    return v;
}

static Value *val_pow_s(Value *a, double p) {
    Value *v = arena_alloc();
    v->data = pow(a->data, p);
    v->n_children = 1;
    v->children[0] = a;
    v->local_grads[0] = p * pow(a->data, p - 1);
    return v;
}

static Value *val_log(Value *a) {
    Value *v = arena_alloc();
    v->data = log(a->data);
    v->n_children = 1; v->children[0] = a;
    v->local_grads[0] = 1.0 / a->data;
    return v;
}

static Value *val_exp(Value *a) {
    Value *v = arena_alloc();
    v->data = exp(a->data);
    v->n_children = 1; v->children[0] = a;
    v->local_grads[0] = v->data;
    return v;
}

static Value *val_relu(Value *a) {
    Value *v = arena_alloc();
    v->data = a->data > 0 ? a->data : 0;
    v->n_children = 1; v->children[0] = a;
    v->local_grads[0] = a->data > 0 ? 1.0 : 0.0;
    return v;
}

static Value *val_neg(Value *a)              { return val_mul(a, val_new(-1.0)); }
static Value *val_sub(Value *a, Value *b)    { return val_add(a, val_neg(b)); }
static Value *val_div(Value *a, Value *b)    { return val_mul(a, val_pow_s(b, -1.0)); }

/* ---- Backward pass (iterative topological sort) ---- */

typedef struct { Value *node; int child_idx; } StackFrame;

static int topo_epoch = 0;
static Value **topo_order = NULL;
static StackFrame *topo_stack = NULL;
static int topo_cap = 0;

static void backward(Value *loss) {
    topo_epoch++;

    int needed = arena_count + 16384;
    if (needed > topo_cap) {
        topo_cap = needed * 2;
        topo_order = (Value **)xrealloc(topo_order, sizeof(Value *) * (size_t)topo_cap, "backward topo_order");
        topo_stack = (StackFrame *)xrealloc(topo_stack, sizeof(StackFrame) * (size_t)topo_cap, "backward topo_stack");
    }

    int topo_count = 0, stack_top = 0;
    topo_stack[stack_top++] = (StackFrame){loss, 0};
    loss->visited = topo_epoch;

    while (stack_top > 0) {
        StackFrame *f = &topo_stack[stack_top - 1];
        if (f->child_idx < f->node->n_children) {
            Value *child = f->node->children[f->child_idx++];
            if (child->visited != topo_epoch) {
                child->visited = topo_epoch;
                topo_stack[stack_top++] = (StackFrame){child, 0};
            }
        } else {
            topo_order[topo_count++] = f->node;
            stack_top--;
        }
    }

    loss->grad = 1.0;
    for (int i = topo_count - 1; i >= 0; i--) {
        Value *v = topo_order[i];
        for (int j = 0; j < v->n_children; j++)
            v->children[j]->grad += v->local_grads[j] * v->grad;
    }
}

/* ======================================================================
 * Neural network primitives (autograd version for training)
 * ====================================================================== */

static void nn_linear(Value **x, Value **w, int nout, int nin, Value **out) {
    for (int i = 0; i < nout; i++) {
        out[i] = val_mul(w[i * nin], x[0]);
        for (int j = 1; j < nin; j++)
            out[i] = val_add(out[i], val_mul(w[i * nin + j], x[j]));
    }
}

static void nn_softmax(Value **logits, int n, Value **out, Value **exps_buf) {
    double max_val = -1e30;
    for (int i = 0; i < n; i++)
        if (logits[i]->data > max_val) max_val = logits[i]->data;

    Value *total = val_new(0);
    for (int i = 0; i < n; i++) {
        exps_buf[i] = val_exp(val_sub(logits[i], val_new(max_val)));
        total = val_add(total, exps_buf[i]);
    }
    for (int i = 0; i < n; i++)
        out[i] = val_div(exps_buf[i], total);
}

static void nn_rmsnorm(Value **x, int n, Value **out) {
    Value *ms = val_mul(x[0], x[0]);
    for (int i = 1; i < n; i++)
        ms = val_add(ms, val_mul(x[i], x[i]));
    ms = val_div(ms, val_new((double)n));
    Value *scale = val_pow_s(val_add(ms, val_new(1e-5)), -0.5);
    for (int i = 0; i < n; i++)
        out[i] = val_mul(x[i], scale);
}

/* ======================================================================
 * Model state: parameter matrices
 * ====================================================================== */

static Value **wte;     /* [vocab_size * N_EMBD] */
static Value **wpe;     /* [BLOCK_SIZE * N_EMBD] */
static Value **lm_head; /* [vocab_size * N_EMBD] */

typedef struct {
    Value **attn_wq; /* [N_EMBD * N_EMBD] */
    Value **attn_wk;
    Value **attn_wv;
    Value **attn_wo;
    Value **mlp_fc1; /* [MLP_DIM * N_EMBD] */
    Value **mlp_fc2; /* [N_EMBD * MLP_DIM] */
} Layer;

static Layer layers[N_LAYER];

static Value **all_params;
static int num_params;

static Value **make_matrix(int nout, int nin, double std) {
    Value **mat = (Value **)xmalloc(sizeof(Value *) * (size_t)nout * (size_t)nin, "make_matrix");
    for (int i = 0; i < nout * nin; i++)
        mat[i] = param_new(rand_gauss(0, std));
    return mat;
}

static void collect_params(Value **mat, int size) {
    for (int i = 0; i < size; i++)
        all_params[num_params++] = mat[i];
}

static void init_model(int vocab_size) {
    int total = vocab_size * N_EMBD
              + BLOCK_SIZE * N_EMBD
              + vocab_size * N_EMBD
              + N_LAYER * (4 * N_EMBD * N_EMBD + MLP_DIM * N_EMBD + N_EMBD * MLP_DIM);
    all_params = (Value **)xmalloc(sizeof(Value *) * (size_t)total, "init_model all_params");
    num_params = 0;

    wte     = make_matrix(vocab_size, N_EMBD, 0.02);
    wpe     = make_matrix(BLOCK_SIZE, N_EMBD, 0.02);
    lm_head = make_matrix(vocab_size, N_EMBD, 0.02);
    collect_params(wte, vocab_size * N_EMBD);
    collect_params(wpe, BLOCK_SIZE * N_EMBD);
    collect_params(lm_head, vocab_size * N_EMBD);

    for (int i = 0; i < N_LAYER; i++) {
        layers[i].attn_wq = make_matrix(N_EMBD, N_EMBD, 0.02);
        layers[i].attn_wk = make_matrix(N_EMBD, N_EMBD, 0.02);
        layers[i].attn_wv = make_matrix(N_EMBD, N_EMBD, 0.02);
        layers[i].attn_wo = make_matrix(N_EMBD, N_EMBD, 0.0);
        layers[i].mlp_fc1 = make_matrix(MLP_DIM, N_EMBD, 0.02);
        layers[i].mlp_fc2 = make_matrix(N_EMBD, MLP_DIM, 0.0);
        collect_params(layers[i].attn_wq, N_EMBD * N_EMBD);
        collect_params(layers[i].attn_wk, N_EMBD * N_EMBD);
        collect_params(layers[i].attn_wv, N_EMBD * N_EMBD);
        collect_params(layers[i].attn_wo, N_EMBD * N_EMBD);
        collect_params(layers[i].mlp_fc1, MLP_DIM * N_EMBD);
        collect_params(layers[i].mlp_fc2, N_EMBD * MLP_DIM);
    }
}

/* ======================================================================
 * GPT forward pass (autograd, for training)
 * ====================================================================== */

static Value *kv_keys[N_LAYER][BLOCK_SIZE][N_EMBD];
static Value *kv_values[N_LAYER][BLOCK_SIZE][N_EMBD];
static int kv_len[N_LAYER];

static void kv_reset(void) {
    for (int i = 0; i < N_LAYER; i++) kv_len[i] = 0;
}

static void gpt_forward(int token_id, int pos_id, int vocab_size, Value **logits_out) {
    Value *x[N_EMBD];

    for (int i = 0; i < N_EMBD; i++)
        x[i] = val_add(wte[token_id * N_EMBD + i], wpe[pos_id * N_EMBD + i]);

    nn_rmsnorm(x, N_EMBD, x);

    for (int li = 0; li < N_LAYER; li++) {
        Value *x_res[N_EMBD];
        for (int i = 0; i < N_EMBD; i++) x_res[i] = x[i];

        Value *xn[N_EMBD];
        nn_rmsnorm(x, N_EMBD, xn);

        Value *q[N_EMBD], *k[N_EMBD], *v[N_EMBD];
        nn_linear(xn, layers[li].attn_wq, N_EMBD, N_EMBD, q);
        nn_linear(xn, layers[li].attn_wk, N_EMBD, N_EMBD, k);
        nn_linear(xn, layers[li].attn_wv, N_EMBD, N_EMBD, v);

        int t = kv_len[li];
        for (int i = 0; i < N_EMBD; i++) {
            kv_keys[li][t][i] = k[i];
            kv_values[li][t][i] = v[i];
        }
        kv_len[li]++;
        int n_cached = kv_len[li];

        Value *x_attn[N_EMBD];
        for (int h = 0; h < N_HEAD; h++) {
            int hs = h * HEAD_DIM;

            Value *attn_logits[BLOCK_SIZE];
            for (int tt = 0; tt < n_cached; tt++) {
                attn_logits[tt] = val_mul(q[hs], kv_keys[li][tt][hs]);
                for (int j = 1; j < HEAD_DIM; j++)
                    attn_logits[tt] = val_add(attn_logits[tt],
                                              val_mul(q[hs + j], kv_keys[li][tt][hs + j]));
                attn_logits[tt] = val_mul(attn_logits[tt], val_new(1.0 / sqrt((double)HEAD_DIM)));
            }

            Value *attn_w[BLOCK_SIZE];
            Value *attn_exps[BLOCK_SIZE];
            nn_softmax(attn_logits, n_cached, attn_w, attn_exps);

            for (int j = 0; j < HEAD_DIM; j++) {
                Value *acc = val_mul(attn_w[0], kv_values[li][0][hs + j]);
                for (int tt = 1; tt < n_cached; tt++)
                    acc = val_add(acc, val_mul(attn_w[tt], kv_values[li][tt][hs + j]));
                x_attn[hs + j] = acc;
            }
        }

        Value *attn_out[N_EMBD];
        nn_linear(x_attn, layers[li].attn_wo, N_EMBD, N_EMBD, attn_out);
        for (int i = 0; i < N_EMBD; i++)
            x[i] = val_add(attn_out[i], x_res[i]);

        /* ---- MLP block ---- */
        for (int i = 0; i < N_EMBD; i++) x_res[i] = x[i];
        nn_rmsnorm(x, N_EMBD, xn);

        Value *h1[MLP_DIM];
        nn_linear(xn, layers[li].mlp_fc1, MLP_DIM, N_EMBD, h1);
        for (int i = 0; i < MLP_DIM; i++)
            h1[i] = val_pow_s(val_relu(h1[i]), 2.0);

        Value *h2[N_EMBD];
        nn_linear(h1, layers[li].mlp_fc2, N_EMBD, MLP_DIM, h2);
        for (int i = 0; i < N_EMBD; i++)
            x[i] = val_add(h2[i], x_res[i]);
    }

    nn_linear(x, lm_head, vocab_size, N_EMBD, logits_out);
}

/* ======================================================================
 * Raw-double inference (no autograd overhead)
 *
 * During inference we never call backward(), so building a computation
 * graph is pure waste. This path operates on plain doubles and avoids
 * all Value allocation, pointer chasing, and graph bookkeeping.
 * ====================================================================== */

/* KV cache for inference (raw doubles) */
static double inf_keys[N_LAYER][BLOCK_SIZE][N_EMBD];
static double inf_values[N_LAYER][BLOCK_SIZE][N_EMBD];
static int inf_kv_len[N_LAYER];

static void inf_kv_reset(void) {
    for (int i = 0; i < N_LAYER; i++) inf_kv_len[i] = 0;
}

static void inf_linear(double *x, Value **w, int nout, int nin, double *out) {
    for (int i = 0; i < nout; i++) {
        double sum = 0;
        for (int j = 0; j < nin; j++)
            sum += w[i * nin + j]->data * x[j];
        out[i] = sum;
    }
}

static void inf_softmax(double *logits, int n, double *out) {
    double max_val = -1e30;
    for (int i = 0; i < n; i++)
        if (logits[i] > max_val) max_val = logits[i];
    double total = 0;
    for (int i = 0; i < n; i++) {
        out[i] = exp(logits[i] - max_val);
        total += out[i];
    }
    for (int i = 0; i < n; i++)
        out[i] /= total;
}

static void inf_rmsnorm(double *x, int n, double *out) {
    double ms = 0;
    for (int i = 0; i < n; i++) ms += x[i] * x[i];
    ms /= n;
    double scale = 1.0 / sqrt(ms + 1e-5);
    for (int i = 0; i < n; i++) out[i] = x[i] * scale;
}

static void inf_gpt_forward(int token_id, int pos_id, int vocab_size, double *logits_out) {
    double x[N_EMBD];

    for (int i = 0; i < N_EMBD; i++)
        x[i] = wte[token_id * N_EMBD + i]->data + wpe[pos_id * N_EMBD + i]->data;

    inf_rmsnorm(x, N_EMBD, x);

    for (int li = 0; li < N_LAYER; li++) {
        double x_res[N_EMBD];
        for (int i = 0; i < N_EMBD; i++) x_res[i] = x[i];

        double xn[N_EMBD];
        inf_rmsnorm(x, N_EMBD, xn);

        double q[N_EMBD], k[N_EMBD], v[N_EMBD];
        inf_linear(xn, layers[li].attn_wq, N_EMBD, N_EMBD, q);
        inf_linear(xn, layers[li].attn_wk, N_EMBD, N_EMBD, k);
        inf_linear(xn, layers[li].attn_wv, N_EMBD, N_EMBD, v);

        int t = inf_kv_len[li];
        for (int i = 0; i < N_EMBD; i++) {
            inf_keys[li][t][i] = k[i];
            inf_values[li][t][i] = v[i];
        }
        inf_kv_len[li]++;
        int n_cached = inf_kv_len[li];

        double x_attn[N_EMBD];
        for (int h = 0; h < N_HEAD; h++) {
            int hs = h * HEAD_DIM;

            double attn_logits[BLOCK_SIZE];
            for (int tt = 0; tt < n_cached; tt++) {
                double dot = 0;
                for (int j = 0; j < HEAD_DIM; j++)
                    dot += q[hs + j] * inf_keys[li][tt][hs + j];
                attn_logits[tt] = dot / sqrt((double)HEAD_DIM);
            }

            double attn_w[BLOCK_SIZE];
            inf_softmax(attn_logits, n_cached, attn_w);

            for (int j = 0; j < HEAD_DIM; j++) {
                double acc = 0;
                for (int tt = 0; tt < n_cached; tt++)
                    acc += attn_w[tt] * inf_values[li][tt][hs + j];
                x_attn[hs + j] = acc;
            }
        }

        double attn_out[N_EMBD];
        inf_linear(x_attn, layers[li].attn_wo, N_EMBD, N_EMBD, attn_out);
        for (int i = 0; i < N_EMBD; i++)
            x[i] = attn_out[i] + x_res[i];

        /* MLP */
        for (int i = 0; i < N_EMBD; i++) x_res[i] = x[i];
        inf_rmsnorm(x, N_EMBD, xn);

        double h1[MLP_DIM];
        inf_linear(xn, layers[li].mlp_fc1, MLP_DIM, N_EMBD, h1);
        for (int i = 0; i < MLP_DIM; i++) {
            double r = h1[i] > 0 ? h1[i] : 0; /* relu */
            h1[i] = r * r;                     /* squared */
        }

        double h2[N_EMBD];
        inf_linear(h1, layers[li].mlp_fc2, N_EMBD, MLP_DIM, h2);
        for (int i = 0; i < N_EMBD; i++)
            x[i] = h2[i] + x_res[i];
    }

    inf_linear(x, lm_head, vocab_size, N_EMBD, logits_out);
}

/* ======================================================================
 * Main: data loading, training, inference
 * ====================================================================== */

int main(void) {

    /* ---- Load dataset ---- */
    FILE *fp = fopen("input.txt", "r");
    if (!fp) {
        fprintf(stderr, "Error: input.txt not found.\n"
                "Download it with:\n"
                "  curl -o input.txt https://raw.githubusercontent.com/karpathy/makemore/"
                "refs/heads/master/names.txt\n");
        return 1;
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        fprintf(stderr, "Error seeking input.txt\n");
        return 1;
    }
    long file_size_long = ftell(fp);
    if (file_size_long < 0) {
        fprintf(stderr, "Error sizing input.txt\n");
        return 1;
    }
    if (fseek(fp, 0, SEEK_SET) != 0) {
        fprintf(stderr, "Error seeking input.txt\n");
        return 1;
    }
    const size_t file_size = (size_t)file_size_long;
    char *file_buf = (char *)xmalloc(file_size + 1, "input buffer");
    if (fread(file_buf, 1, file_size, fp) != file_size) {
        fprintf(stderr, "Error reading input.txt\n");
        return 1;
    }
    file_buf[file_size] = '\0';
    fclose(fp);

    /* Parse documents (one per line) */
    char *doc_ptrs[MAX_DOCS];
    int doc_lens[MAX_DOCS];
    int num_docs = 0;
    char *line = file_buf;
    char *file_end = file_buf + file_size;
    while (line < file_end && num_docs < MAX_DOCS) {
        char *line_end = line;
        while (line_end < file_end && *line_end != '\n')
            line_end++;
        if (line_end < file_end)
            *line_end = '\0';

        /* trim whitespace */
        while (*line == ' ' || *line == '\r') line++;
        int len = (int)strlen(line);
        while (len > 0 && (line[len-1] == ' ' || line[len-1] == '\r')) len--;
        line[len] = '\0';

        if (len > 0) {
            doc_ptrs[num_docs] = line;
            doc_lens[num_docs] = len;
            num_docs++;
        }

        if (line_end >= file_end) break;
        line = line_end + 1;
    }

    int doc_order[MAX_DOCS];
    for (int i = 0; i < num_docs; i++) doc_order[i] = i;
    shuffle_ints(doc_order, num_docs);
    printf("num docs: %d\n", num_docs);

    /* ---- Build vocabulary from unique characters ---- */
    int char_seen[256] = {0};
    for (int d = 0; d < num_docs; d++)
        for (int i = 0; i < doc_lens[d]; i++)
            char_seen[(unsigned char)doc_ptrs[d][i]] = 1;

    char uchars[256];
    int num_uchars = 0;
    for (int c = 0; c < 256; c++)
        if (char_seen[c]) uchars[num_uchars++] = (char)c;
    /* uchars is sorted since we iterate 0..255 */

    /* O(1) char-to-token-id lookup (replaces the O(vocab_size) linear scan) */
    int char_to_id[256];
    memset(char_to_id, 0, sizeof(char_to_id));
    for (int i = 0; i < num_uchars; i++)
        char_to_id[(unsigned char)uchars[i]] = i;

    int BOS = num_uchars;
    int vocab_size = num_uchars + 1;
    printf("vocab size: %d\n", vocab_size);

    /* ---- Initialize model ---- */
    arena_init(500000);
    init_model(vocab_size);
    printf("num params: %d\n", num_params);

    /* ---- Adam optimizer buffers ---- */
    double learning_rate = 1e-2, beta1 = 0.9, beta2 = 0.95, eps_adam = 1e-8;
    double *adam_m = (double *)xcalloc((size_t)num_params, sizeof(double), "adam_m");
    double *adam_v = (double *)xcalloc((size_t)num_params, sizeof(double), "adam_v");
    Value **logits_buf = (Value **)xmalloc(sizeof(Value *) * (size_t)vocab_size, "train logits_buf");
    Value **probs_buf = (Value **)xmalloc(sizeof(Value *) * (size_t)vocab_size, "train probs_buf");
    Value **softmax_exps_buf = (Value **)xmalloc(sizeof(Value *) * (size_t)vocab_size, "train softmax_exps_buf");

    /* ---- Training loop ---- */
    int num_steps = 500;

    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    for (int step = 0; step < num_steps; step++) {

        int di = doc_order[step % num_docs];
        char *doc = doc_ptrs[di];
        int dlen = doc_lens[di];

        /* Tokenize on the fly: [BOS] + char_ids + [BOS] */
        int tokens[MAX_DOC_LEN + 2];
        tokens[0] = BOS;
        int actual_dlen = dlen < MAX_DOC_LEN ? dlen : MAX_DOC_LEN;
        for (int i = 0; i < actual_dlen; i++)
            tokens[i + 1] = char_to_id[(unsigned char)doc[i]];
        tokens[actual_dlen + 1] = BOS;
        int seq_len = actual_dlen + 2;
        int n = BLOCK_SIZE < (seq_len - 1) ? BLOCK_SIZE : (seq_len - 1);

        arena_reset();
        kv_reset();

        Value *losses[BLOCK_SIZE];

        for (int pos = 0; pos < n; pos++) {
            int token_id = tokens[pos];
            int target_id = tokens[pos + 1];

            gpt_forward(token_id, pos, vocab_size, logits_buf);
            nn_softmax(logits_buf, vocab_size, probs_buf, softmax_exps_buf);
            losses[pos] = val_neg(val_log(probs_buf[target_id]));
        }

        Value *loss = losses[0];
        for (int i = 1; i < n; i++)
            loss = val_add(loss, losses[i]);
        loss = val_mul(loss, val_new(1.0 / n));

        backward(loss);

        double lr_t = learning_rate * 0.5 * (1 + cos(M_PI * step / (double)num_steps));
        const double beta1_correction = 1.0 - pow(beta1, step + 1);
        const double beta2_correction = 1.0 - pow(beta2, step + 1);
        for (int i = 0; i < num_params; i++) {
            Value *p = all_params[i];
            adam_m[i] = beta1 * adam_m[i] + (1 - beta1) * p->grad;
            adam_v[i] = beta2 * adam_v[i] + (1 - beta2) * p->grad * p->grad;
            double m_hat = adam_m[i] / beta1_correction;
            double v_hat = adam_v[i] / beta2_correction;
            p->data -= lr_t * m_hat / (sqrt(v_hat) + eps_adam);
            p->grad = 0;
        }

        printf("step %4d / %4d | loss %.4f\n", step + 1, num_steps, loss->data);
    }

    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    double train_secs = (ts_end.tv_sec - ts_start.tv_sec)
                      + (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;
    printf("\ntraining time: %.3f seconds\n", train_secs);

    /* ---- Inference (raw doubles, no autograd) ---- */
    double temperature = 0.5;
    double *inf_logits = (double *)xmalloc(sizeof(double) * (size_t)vocab_size, "inference logits");
    double *inf_probs = (double *)xmalloc(sizeof(double) * (size_t)vocab_size, "inference probs");
    printf("\n--- inference ---\n");

    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    for (int s = 0; s < 20; s++) {
        inf_kv_reset();

        int token_id = BOS;
        char sample[BLOCK_SIZE + 1];
        int sample_len = 0;

        for (int pos = 0; pos < BLOCK_SIZE; pos++) {
            inf_gpt_forward(token_id, pos, vocab_size, inf_logits);

            for (int i = 0; i < vocab_size; i++)
                inf_logits[i] /= temperature;
            inf_softmax(inf_logits, vocab_size, inf_probs);

            token_id = weighted_choice(inf_probs, vocab_size);
            if (token_id == BOS) break;
            sample[sample_len++] = uchars[token_id];
        }
        sample[sample_len] = '\0';
        printf("sample %2d: %s\n", s + 1, sample);
    }

    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    double inf_secs = (ts_end.tv_sec - ts_start.tv_sec)
                    + (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;
    printf("\ninference time: %.6f seconds\n", inf_secs);

    /* Cleanup (just the heap allocations that aren't OS-reclaimed-on-exit ceremony) */
    free(file_buf);
    free(adam_m);
    free(adam_v);
    free(logits_buf);
    free(probs_buf);
    free(softmax_exps_buf);
    free(inf_logits);
    free(inf_probs);
    free(arena_buf);
    free(topo_order);
    free(topo_stack);

    return 0;
}
