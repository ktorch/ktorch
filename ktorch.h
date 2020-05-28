#pragma once

#ifdef __clang__
# pragma clang diagnostic push
# pragma GCC diagnostic ignored "-Wgnu-anonymous-struct"                   // k.h warning
# pragma GCC diagnostic ignored "-Wnested-anon-types"                      // k.h warning
# pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"    // ATen.h VA_ARG warning, FORWARD_HAS_DEFAULT_ARGS
# pragma clang diagnostic ignored "-Wunused-function"                      // private.h generates 'unused function' warnings
#elif defined __GNUC__
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wpedantic"
# pragma GCC diagnostic ignored "-Wunused-function"
#endif

#define KXVER 3
#include "k.h"
#undef P
#undef R
#undef U
#undef Z
#undef xs

#include <stack>
#include "torch/torch.h"
#include "knn.h"
#include "private.h"

// access private name_ element of torch::nn::Module
ACCESS_PRIVATE_FIELD(torch::nn::Module, c10::optional<std::string>, name_)

// template to make overloaded fn for use with variants and visit
template <class... Fs> struct overload;

template <class F0, class... Frest> struct overload<F0, Frest...> : F0, overload<Frest...> {
    overload(F0 f0, Frest... rest) : F0(f0), overload<Frest...>(rest...) {}
    using F0::operator();
    using overload<Frest...>::operator();
};
template <class F0> struct overload<F0> : F0 {
    overload(F0 f0) : F0(f0) {}
    using F0::operator();
};
template <class... Fs> auto make_overload(Fs... fs) {
    return overload<Fs...>(fs...);
}

#ifdef __clang__
# pragma clang diagnostic pop
#elif defined __GNUC__
# pragma GCC diagnostic pop
#endif

#define KFN(f) reinterpret_cast<void *>(f)
#define KERR(e) krr((S)e)

#define KTRY \
 try {
#define KCATCH(x)                                                           \
 } catch (const c10::Error &e) {                                            \
  return KERR(krrbuf(env().frame ? e.what() : e.what_without_backtrace())); \
 } catch (const std::exception &e) {                                        \
  return KERR(krrbuf(e.what()));                                            \
 } catch (...) {                                                            \
  return KERR(x);                                                           \
 }

#ifdef __cplusplus
# define KEXT extern "C"
#else
# define KEXT
#endif

#ifdef _WIN32
# define KAPI KEXT __declspec(dllexport) K
#else
# define KAPI KEXT K
#endif

#define Ksize torch::SmallVector<int64_t,8>
#define cs(x) ss((S)x)

using Ktype=signed char;
using Storage=torch::Storage;
using Tensor=torch::Tensor;
using Scalar=torch::Scalar;
using TensorVector=std::vector<Tensor>;
using TensorDeque=std::deque<Tensor>;
using LongVector=std::vector<int64_t>;
using IntArrayRef=torch::IntArrayRef;
template<size_t D,typename T=int64_t> using ExpandingArray=torch::ExpandingArray<D,T>;
template<size_t D,typename T=double>  using Exdouble=torch::ExpandingArray<D,T>;
template<size_t D,typename T=int64_t> using Exoptional=torch::ExpandingArrayWithOptionalElem<D,T>;
using ScalarType=torch::ScalarType;
using TypeMeta=caffe2::TypeMeta;
using TensorOptions=torch::TensorOptions;
using TensorList=torch::TensorList;

using Optimizer=torch::optim::Optimizer;
using Optptr=std::shared_ptr<Optimizer>;
using TensorDict = torch::OrderedDict<std::string, torch::Tensor>;

// shorter names for commononly used container modules defined by pytorch & created in knn.h
using Module=torch::nn::Module;
using AnyModule=torch::nn::AnyModule;
using Sequential=torch::nn::Sequential;
class SeqNest;
class SeqJoin;

// define a kind of union for modules used to build sequences
using Layer=c10::variant<Sequential, SeqNest, SeqJoin, AnyModule>;
enum class Layers       {sequential, seqnest, seqjoin, any};
using Layerstack=std::stack<Layer>;

typedef struct {
 Ktype a = 0;  // type: 1-dict, 2-list of pairs, 3-general list, 4-sym list
 Ktype t = 0;  // type of value in last pair processed
 H i = 0;  // next pair to process
 H n = 0;  // count of pairs
 S k = 0;  // name of an evaluated name,value pair
 K x = 0;  // k value with dict/pairs/list
 union {
  bool   b;  // boolean value from last evaluated pair
  J      j;  // long value
  double f;  // double value
  S      s;  // symbol value
  K      v;  // value (isn't sym or numeric scalar)
 };
} Pairs;

enum class Class:char {
 undefined=0,
 tensor,
 vector,
 module,
 layer,
 sequential,
 loss,
 optimizer,
 model
};

enum class Cast:char {
 undefined=0, 
 tensor,  model,                    // basic structures
 sequential, seqnest, seqjoin,      // container modules
 any,        anyname,               // generic modules

 adaptavg1d,     adaptavg2d,      adaptavg3d,      adaptmax1d,      adaptmax2d,  // modules
 adaptmax3d,     adrop,           attention,       avgpool1d,       avgpool2d,
 avgpool3d,      batchnorm1d,     batchnorm2d,     batchnorm3d,
 bilinear,       cat,             celu,            conv1d,          conv2d,
 conv3d,         convtranspose1d, convtranspose2d, convtranspose3d, crossmap2d,
 drop,           drop2d,          drop3d,          elu,             embed,
 embedbag,       expand,          fadrop,          flatten,
 fmaxpool2d,     fmaxpool3d,      fold,            gelu,            glu,
 groupnorm,      gru,             hardshrink,      hardtanh,        identity,
 instancenorm1d, instancenorm2d,  instancenorm3d,  layernorm,       leakyrelu,
 linear,         localnorm,       logsigmoid,      logsoftmax,      lppool1d,
 lppool2d,       lstm,            maxpool1d,       maxpool2d,       maxpool3d,
 normalize,      pad,             pad1d,           pad2d,           pad3d,
 prelu,          reflect1d,       reflect2d,       relu,            relu6,
 replicate1d,    replicate2d,     replicate3d,     reshape,         rnn,
 rrelu,          selu,            sigmoid,         softmax,         softmax2d,
 softmin,        softplus,        softshrink,      softsign,        squeeze,
 tanh,           tanhshrink,      threshold,       unfold,          unsqueeze,
 zeropad2d,

 pairwise,  similar, // distance functions

 bce,       bcelogits, ce,          cosineloss, ctc,        hinge,        //loss fns
 kl,        l1,        margin,      mse,        multilabel, multimargin,
 multisoft, nll,       poissonloss, smoothl1,   softmargin, triplet,    

 adagrad, adam, lbfgs, rmsprop, sgd //optimizers
};

enum class Tensormode:char {   // tensor creation modes
 undefined,
 arange, empty, eye,     full,  linspace, logspace,
 ones,   rand,  randint, randn, randperm, range,
 zeros
};

enum class Prob:char {  // probablility distributions
 cauchy, exponential, geometric, lognormal, normal, random, uniform
};

enum class Setting:char {
 undefined,
 affine,  alpha,     amsgrad,   batchfirst, beta,      beta1,      beta2,
 bi,      bias,      blank,     ceiling,    centered,  changetol,  channels,
 cols,    countpad,  dampening, decay,      dilate,    dim,        divisor,
 dropout, end,       eps,       eval,       fn,        freeze,     full,
 gradtol, groups,    heads,     hidden,     history,   ignore,     in,
 in1,     in2,       indices,   init,       inplace,   iter,       k,
 kdim,    keepdim,   kvbias,    kvzeros,    lambda,    lastoffset, layers,
 log,     lower,     lr,        lrdecay,    margin,    max,        maxnorm,
 min,     mode,      momentum,  nesterov,   out,       outpad,     outsize,
 p,       pad,       padindex,  padmode,    ratio,     reduce,     rows,
 scale,   shape,     size,      slope,      sparse,    start,      stride,
 swap,    threshold, track,     train,      transpose, type,       upper,
 value,   vdim,      weight,    zeroinf
};

enum class State:char {
 Class,depth,module,name,options,parms,buffers
};

enum class Attr:char {
 undefined = 0,
 dim, elementsize, numel,  offset, ptr, ref, sparsedim, weakref, // long scalars
 device, dtype, gradfn, gradient, layout,                        // symbol
 coalesced, contiguous, leaf, pinned,                            // boolean
 size, stride,                                                   // long list
 data, storage                                                   // other: list,dict,..
};
 
enum class Metric: char {
 loss, accuracy, max, out
};

enum class Enum {  // enums to match pytorch variants
 undefined=-1,
 area,            batchmean,       bicubic,         bilinear,  border,   
 circular,        constant,        conv1d,          conv2d,    conv3d,   
 convtranspose1d, convtranspose2d, convtranspose3d, fanin,     fanout,   
 leakyrelu,       linear,          max,             mean,      nearest,  
 none,            reflect,         reflection,      relu,      replicate,
 sigmoid,         sum,             tanh,            trilinear, zeros,    
};

struct TORCH_API Ktag {
 Class a = Class::undefined;
 Cast  c = Cast::undefined;
 virtual ~Ktag() = default;
};

struct TORCH_API Kten : public Ktag {
 Tensor t;
 Kten(const Tensor& x) : t(std::move(x)) {a=Class::tensor; c=Cast::tensor;}
};

struct TORCH_API Kvec : public Ktag {
 TensorVector v;
 Kvec(const TensorVector& x) : v(std::move(x)) {a=Class::vector; c=Cast::tensor;}
};

struct TORCH_API Klayer : public Ktag {
 Layer m;       // name of single module or container of many modules, e.g. Sequential
 Klayer(Cast x,const Layer& y) : m(std::move(y)) {a=Class::layer; c=x;}
};

struct TORCH_API Kmodule : public Ktag {
 Kmodule(Class x,Cast y,const AnyModule& z) : m(std::move(z)) {a=x; c=y;}
 AnyModule m;
};

struct TORCH_API Kopt : public Ktag {
 Optptr o;
 Kopt(Cast x,const Optptr& y) : o(std::move(y)) {a=Class::optimizer; c=x;}
 bool is_empty() const noexcept {return o == nullptr;}
 Optimizer* get() {TORCH_CHECK(!is_empty(), "undefined optimizer"); return o.get();}
};

struct TORCH_API Kmodel : public Ktag {
 Cast mc;          // type of module, typically a container module, e.g. Sequential
 Cast lc;          // type of loss fn
 Cast oc;          // type of optimizer
 Layer m;          // layer(s), e.g. Sequential
 AnyModule l;      // loss module
 Optptr o;         // shared ptr to optimizer
 Kmodel(Klayer *x,Kmodule *y,Kopt *z) : mc(x->c),lc(y->c),oc(z->c),m(x->m),l(y->m),o(z->o) {a=Class::model; c=Cast::model;}
};

S krrbuf(const char *);
void dictadd(K,S,K);
void dictadd(K,const char*,K);
bool xind(K,J);
bool xind(K,J,Ktype);
K kptr(void*);
bool xptr(K);
bool xptr(K,J);
Ktag* xtag(K);
Ktag* xtag(K,J);

bool null(const char*);
bool null(const J);
bool match(const Scalar&,const Scalar&);
K kscalar(const Scalar&);

J xlen(K);
J xlen(K,J);
const char* kname(Ktype);
const char* kname(K);
const char* kname(K,J);

J ksizeof(Ktype);
Ktype maptype(TypeMeta);
TypeMeta maptype(Ktype);
S mapclass(Class);
S mapattr(Attr);
Enum emap(S);

S statekey(State);
J statefind(State,K);
J statedepth(K x,J j=-1);
S statemodule(K x,J j=-1);
S statename(K x,J j=-1);
K stateoptions(K x,J j=-1);
K stateparms(K x,J j=-1);
K statebuffers(K x,J j=-1);
void stateparms(S,Module&,K,bool);

bool xnull(K);
bool xnull(K,J);
bool xempty(K);
bool xempty(K,J);
bool xmixed(K,J);
bool xsym(K);
bool xsym(K,J);
bool xsym(K,S&);
bool xsym(K,J,S&);
bool xsyms(K,S&);
bool xdev(K,torch::Device&);
bool xdev(K,J,torch::Device&);

bool xint64(K,int64_t&);
bool xint64(K,J,int64_t&);
bool xlong(K,J&);
bool xlong(K,J,J&);
bool xlong(K,J&,J*&);
bool xlong(K,J,J&,J*&);
bool xdouble(K,double&);
bool xdouble(K,J,double&);
bool xdouble(K,J&,double *&);
bool xdouble(K,J,J&,double *&);

bool xdict(K);
bool xdict(K,J);
bool xstate(K);
bool xstate(K,J);

bool xsize(K,IntArrayRef&);
bool xsize(K,J,IntArrayRef&);
bool xsize(K,J,int64_t*);
bool xsize(K,J,double*);
bool xsize(K,J,J,int64_t*);
bool xsize(K,J,J,double*);

bool xten(K,Tensor&);
bool xten(K,J,Tensor&);
Tensor* xten(K);
Tensor* xten(K,J);
bool xtenpair(K,Tensor&,Tensor&);
bool xtenpair(K,J,Tensor&,Tensor&);
bool xten3(K,Tensor&,Tensor&,Tensor&);
bool xten3(K,J,Tensor&,Tensor&,Tensor&);
bool xtenarg(K,J,Tensor&,Tensor&);
bool xtenarg(K,J,Tensor&,Tensor&,Tensor&);
bool xtenarg(K,Tensor&,Tensor&);
bool xtenarg(K,Tensor&,Tensor&,Tensor&);

Kmodule* xmodule(K);
Kmodule* xmodule(K,J);
Klayer* xlayer(K);
Klayer* xlayer(K,J);
Kmodule* xloss(K);
Kmodule* xloss(K,J);
Kopt* xoptim(K);
Kopt* xoptim(K,J);
Kmodel* xmodel(K);
Kmodel* xmodel(K,J);
TensorVector* xvec(K);
TensorVector* xvec(K,J);

bool xnum(K,double&);
bool xnum(K,J,double&);
bool xnum(K,Scalar&);
bool xnum(K,J,Scalar&);
bool xnumn(K,c10::optional<Scalar>&);
bool xnumn(K,J,c10::optional<Scalar>&);
bool xnumt(K,Scalar&);
bool xnumt(K,J,Scalar&);
bool xnumlist(K,J,Scalar&);
bool xbyte(K,Scalar&);
bool xbyte(K,J,Scalar&);
bool xscalar(K,Scalar&);
bool xscalar(K,J,Scalar&);

bool xbool(K,bool&);
bool xbool(K,J,bool&);
TypeMeta mtype(S);
S mtype(TypeMeta);
ScalarType stype(S);
S stype(ScalarType);
S stype(c10::optional<ScalarType>);
bool xtype(K,ScalarType&);
bool xtype(K,J,ScalarType&);
bool xtype(K,c10::optional<ScalarType>&);
bool xtype(K,J,c10::optional<ScalarType>&);
bool xtype(K,TypeMeta&);
bool xtype(K,J,TypeMeta&);
bool xopt(S,TensorOptions&);
bool xopt(K,TensorOptions&);
bool xopt(K,J,TensorOptions&);
bool xto(S,TensorOptions&);
bool xto(K,TensorOptions&);
bool xto(K,J,TensorOptions&);
bool xmode(K,S&,Tensormode&);
bool xmode(K,J,S&,Tensormode&);
bool xbacksym(K,bool&,bool&);
bool xbacksym(K,J,bool&,bool&);

bool xpairs(K,Pairs&);
bool xpairs(K,J,Pairs&);
bool xpair(Pairs&);
J xargc(K,J,Pairs&);
bool xnone(K,J);

S psym(const Pairs&);
ScalarType ptype(const Pairs&);
void perr(const Pairs&,const char*);
bool pempty(const Pairs&);
bool pbool(const Pairs&);
J plong(const Pairs&);
double pdouble(const Pairs&);
void pnum(const Pairs&,Scalar&);
void psize(const Pairs&,IntArrayRef&,J n=-1);
void psize(const Pairs&,J,int64_t*);
void psize(const Pairs&,J,double*);
void pten(const Pairs&,Tensor&);

S& optsym(const torch::Device&);
S& optsym(const TypeMeta&);
S& optsym(const torch::Layout&);
S& optsym(const bool&);
K optkey();
K optval(const TensorOptions &o,K x,J i=-1);
K optmap(const TensorOptions&);
std::string kstring(K);
K kout(K);
K kcast(Ktype,K);
K kbool(K);
K kdict(const TensorDict&);
J kfind(K,const std::string&);
K klist(J,const int64_t*);
K klist(J,const double*);
K klist(J,const c10::optional<int64_t>*);
K kexpand(J,const int64_t*);
K kexpand(J,const double*);
K kexpand(J,const c10::optional<int64_t>*e);
#define KEX(x) kexpand(x.size(),(*x).data())  // k list from ExpandingArray
J xdv(K);

S objdevice(const Tensor&);
S objdevice(const TensorVector&,S);
bool kfree(K);
bool kfree(K,J);
void fn(K,const char*,void*,I);
void randomfn(K);
void mathfn(K);

// tensor routines:
K kget(const Tensor&);
K kget(const LongVector&);
K kget(const TensorVector&);
K kget(const TensorDeque&);
Tensor kput(K);
Tensor kput(K,J);
K kten(const Tensor&);
K kvec(const TensorVector&);
inline K kresult(bool p,const Tensor& t) {return p ? kten(t) : kget(t);}
K to(Kten*,const TensorOptions&,bool,bool);
K to(Kvec*,const TensorOptions&,bool);
K ktenpair(bool,Tensor&,Tensor&);
K kten3(bool,Tensor&,Tensor&,Tensor&);
J tensorlong(const Tensor&,Attr);
S tensorsym(const Tensor&,Attr);
K tensorsize(const Tensor&,Attr);
K tensorattr(const Tensor&,Ktype,Attr);
K vectorattr(const TensorVector&,Ktype,Attr);
K tensorinfo(const Tensor&,bool);
K vectorinfo(const TensorVector&,bool);
void tensorcopy(Tensor&,const Tensor&,bool async=false);
Tensor       shuffle(const Tensor &t,      int64_t d=0);
TensorVector shuffle(const TensorVector& v,int64_t d=0);
void shuffle_(Tensor &t,      int64_t d=0);
void shuffle_(TensorVector& v,int64_t d=0);
std::vector<int64_t> newsize(const Tensor&,int64_t,int64_t);
int64_t maxsize(const Tensor& t,      int64_t d=0);
int64_t maxsize(const TensorVector& v,int64_t d=0);
int64_t fullsize(Tensor& t,      int64_t d=0);
int64_t fullsize(TensorVector& v,int64_t d=0);
int64_t subsets(int64_t w,int64_t n,bool b=false);
void subset(Tensor& t,int64_t d,int64_t i,int64_t w,int64_t n=0);
void subset(TensorVector& v,int64_t d,int64_t i,int64_t w,int64_t n=0);
void setsafe(Tensor& t,const Storage&,int64_t,const IntArrayRef&,const IntArrayRef&);
K tensorback(K);
void tensorfn(K);

// nn module & functional routines:
// distance module arg set/get (for use in loss functions)
torch::nn::CosineSimilarityOptions  similar(K,J,Cast);
torch::nn::PairwiseDistanceOptions pairwise(K,J,Cast);
void  similar(bool,K,const torch::nn::CosineSimilarityOptions&);
void pairwise(bool,K,const torch::nn::PairwiseDistanceOptions&);


K klayer(Cast,const Layer&);
K to(Klayer*,const TensorOptions&,bool);
Module& mref(const Layer&);
Module& mref(Klayer*);
Module& mref(Kmodel*);
c10::optional<std::string>& mname(Module&);
std::string mlabel(const std::type_info&);
K mget(bool,bool,const char*,const Module&);
K mforward(Layer&,K);
Tensor mforward(Layer& q,const Tensor& x,const Tensor& y={},const Tensor& z={});
K mattr(const Layer&,Ktype,Attr);
void nnfn(K);

// loss functions:
K kloss(Cast,const AnyModule&);
Tensor losswt(Cast,AnyModule&,const Tensor&,const Tensor&);
K lossdict(Ktag*,K);
K lossdict(bool,bool,Cast,AnyModule&);
K to(Kmodule*,const TensorOptions&,bool);
K lossattr(const AnyModule&,Ktype,Attr);
void lossfn(K);

// optimization functions:
K kopt(Cast,const Optptr&);
K optstate(Ktag*,K);
K optstate(bool,bool,Cast,Optimizer*);
K optattr(const Optptr&,Ktype,Attr);
void optstep(Cast,Optptr&);
void optstep(Kopt*);
void optstep(Kmodel*);
void optfn(K);

// model functions:
K modelstate(Ktag*,K);
K mbackward(K);
void modelfn(K);

// global environment
typedef struct {
 I cuda;             // number of CUDA devices
 bool frame=false;      // if true, error message returns stack frame
 bool alloptions=true;  // if true, return all option settings, else only non-defaults
 S help=cs("help");

 std::vector<std::tuple<S,torch::Device>> device;

 std::array<std::tuple<Ktype,TypeMeta>,8> ktype = {{           //k type -> torch type
  std::make_tuple(KE, at::scalarTypeToTypeMeta(at::kFloat)),
  std::make_tuple(KF, at::scalarTypeToTypeMeta(at::kDouble)),
  std::make_tuple(KJ, at::scalarTypeToTypeMeta(at::kLong)),
  std::make_tuple(KI, at::scalarTypeToTypeMeta(at::kInt)),
  std::make_tuple(KH, at::scalarTypeToTypeMeta(at::kShort)),
  std::make_tuple(KB, at::scalarTypeToTypeMeta(at::kBool)),
  std::make_tuple(KG, at::scalarTypeToTypeMeta(at::kByte)),
  std::make_tuple(KC, at::scalarTypeToTypeMeta(at::kChar))
 }};

 std::array<std::tuple<S,TypeMeta,Ktype>,9> dtype = {{       //sym -> torch type -> k type
  std::make_tuple(cs("float"),  at::scalarTypeToTypeMeta(at::kFloat),  KE),
  std::make_tuple(cs("double"), at::scalarTypeToTypeMeta(at::kDouble), KF),
  std::make_tuple(cs("half"),   at::scalarTypeToTypeMeta(at::kHalf),   KE),
  std::make_tuple(cs("bool"),   at::scalarTypeToTypeMeta(at::kBool),   KB),
  std::make_tuple(cs("byte"),   at::scalarTypeToTypeMeta(at::kByte),   KG),
  std::make_tuple(cs("char"),   at::scalarTypeToTypeMeta(at::kChar),   KC),
  std::make_tuple(cs("long"),   at::scalarTypeToTypeMeta(at::kLong),   KJ),
  std::make_tuple(cs("int"),    at::scalarTypeToTypeMeta(at::kInt),    KI),
  std::make_tuple(cs("short"),  at::scalarTypeToTypeMeta(at::kShort),  KH)
 }};

 std::array<std::tuple<S,torch::Layout>,2> layout = {{
  std::make_tuple(cs("strided"),torch::kStrided),          
  std::make_tuple(cs("sparse"), torch::kSparse)
 }};

 std::array<std::tuple<S,bool>,2> gradient = {{
  std::make_tuple(cs("grad"),   true),          
  std::make_tuple(cs("nograd"), false)
 }};

/*
 std::array<std::tuple<S,bool>,2> async = {{
  std::make_tuple(cs("async"),   true),          
  std::make_tuple(cs("sync"),   false)
 }};
*/

 std::array<std::tuple<S,Class>,8> kclass = {{
  std::make_tuple(cs("tensor"),     Class::tensor),          
  std::make_tuple(cs("vector"),     Class::vector),
  std::make_tuple(cs("module"),     Class::module),
  std::make_tuple(cs("layer"),      Class::layer),
  std::make_tuple(cs("sequential"), Class::sequential),
  std::make_tuple(cs("loss"),       Class::loss),
  std::make_tuple(cs("optimizer"),  Class::optimizer),
  std::make_tuple(cs("model"),      Class::model)
 }};

 std::array<std::tuple<S,Class>,3> model = {{
  std::make_tuple(cs("seq"),  Class::sequential),
  std::make_tuple(cs("loss"), Class::loss),
  std::make_tuple(cs("opt"),  Class::optimizer),
 }};

 std::array<std::tuple<S,Tensormode>,13> tensormode = {{    //tensor creation mode: map symbol -> enum
  std::make_tuple(cs("empty"),    Tensormode::empty),
  std::make_tuple(cs("full"),     Tensormode::full),
  std::make_tuple(cs("eye"),      Tensormode::eye),
  std::make_tuple(cs("ones"),     Tensormode::ones),
  std::make_tuple(cs("zeros"),    Tensormode::zeros),
  std::make_tuple(cs("rand"),     Tensormode::rand),
  std::make_tuple(cs("randn"),    Tensormode::randn),
  std::make_tuple(cs("randint"),  Tensormode::randint),
  std::make_tuple(cs("randperm"), Tensormode::randperm),
  std::make_tuple(cs("range"),    Tensormode::range),
  std::make_tuple(cs("arange"),   Tensormode::arange),
  std::make_tuple(cs("linspace"), Tensormode::linspace),
  std::make_tuple(cs("logspace"), Tensormode::logspace)
 }};

 std::array<std::tuple<S,Prob>,7> prob = {{    //probability distribution: map symbol -> enum
  std::make_tuple(cs("cauchy"),      Prob::cauchy),
  std::make_tuple(cs("exponential"), Prob::exponential),
  std::make_tuple(cs("geometric"),   Prob::geometric),
  std::make_tuple(cs("lognormal"),   Prob::lognormal),
  std::make_tuple(cs("normal"),      Prob::normal),
  std::make_tuple(cs("random"),      Prob::random),
  std::make_tuple(cs("uniform"),     Prob::uniform),
 }};

/* PATCH
 std::array<std::tuple<S,torch::nn::RNNActivation>,2> rnnfn = {{
  std::make_tuple(cs("relu"),torch::nn::RNNActivation::ReLU),
  std::make_tuple(cs("tanh"),torch::nn::RNNActivation::Tanh)
 }};
*/

 std::array<std::tuple<S,Cast>,94> module = {{               // module sym -> enum
  std::make_tuple(cs("adaptavg1d"),      Cast::adaptavg1d),
  std::make_tuple(cs("adaptavg2d"),      Cast::adaptavg2d),
  std::make_tuple(cs("adaptavg3d"),      Cast::adaptavg3d),
  std::make_tuple(cs("adaptmax1d"),      Cast::adaptmax1d),
  std::make_tuple(cs("adaptmax2d"),      Cast::adaptmax2d),
  std::make_tuple(cs("adaptmax3d"),      Cast::adaptmax3d),
  std::make_tuple(cs("adrop"),           Cast::adrop),
  std::make_tuple(cs("attention"),       Cast::attention),
  std::make_tuple(cs("avgpool1d"),       Cast::avgpool1d),
  std::make_tuple(cs("avgpool2d"),       Cast::avgpool2d),
  std::make_tuple(cs("avgpool3d"),       Cast::avgpool3d),
  std::make_tuple(cs("batchnorm1d"),     Cast::batchnorm1d),
  std::make_tuple(cs("batchnorm2d"),     Cast::batchnorm2d),
  std::make_tuple(cs("batchnorm3d"),     Cast::batchnorm3d),
  std::make_tuple(cs("bilinear"),        Cast::bilinear),
  std::make_tuple(cs("cat"),             Cast::cat),
  std::make_tuple(cs("celu"),            Cast::celu),
  std::make_tuple(cs("conv1d"),          Cast::conv1d),
  std::make_tuple(cs("conv2d"),          Cast::conv2d),
  std::make_tuple(cs("conv3d"),          Cast::conv3d),
  std::make_tuple(cs("convtranspose1d"), Cast::convtranspose1d),
  std::make_tuple(cs("convtranspose2d"), Cast::convtranspose2d),
  std::make_tuple(cs("convtranspose3d"), Cast::convtranspose3d),
  std::make_tuple(cs("crossmap2d"),      Cast::crossmap2d),
  std::make_tuple(cs("drop"),            Cast::drop),
  std::make_tuple(cs("drop2d"),          Cast::drop2d),
  std::make_tuple(cs("drop3d"),          Cast::drop3d),
  std::make_tuple(cs("elu"),             Cast::elu),
  std::make_tuple(cs("embed"),           Cast::embed),
  std::make_tuple(cs("embedbag"),        Cast::embedbag),
  std::make_tuple(cs("expand"),          Cast::expand),
  std::make_tuple(cs("fadrop"),          Cast::fadrop),
  std::make_tuple(cs("flatten"),         Cast::flatten),
  std::make_tuple(cs("fmaxpool2d"),      Cast::fmaxpool2d),
  std::make_tuple(cs("fmaxpool3d"),      Cast::fmaxpool3d),
  std::make_tuple(cs("fold"),            Cast::fold),
  std::make_tuple(cs("gelu"),            Cast::gelu),
  std::make_tuple(cs("glu"),             Cast::glu),
  std::make_tuple(cs("groupnorm"),       Cast::groupnorm),
  std::make_tuple(cs("gru"),             Cast::gru),
  std::make_tuple(cs("hardshrink"),      Cast::hardshrink),
  std::make_tuple(cs("hardtanh"),        Cast::hardtanh),
  std::make_tuple(cs("identity"),        Cast::identity),
  std::make_tuple(cs("instancenorm1d"),  Cast::instancenorm1d),
  std::make_tuple(cs("instancenorm2d"),  Cast::instancenorm2d),
  std::make_tuple(cs("instancenorm3d"),  Cast::instancenorm3d),
  std::make_tuple(cs("layernorm"),       Cast::layernorm),
  std::make_tuple(cs("leakyrelu"),       Cast::leakyrelu),
  std::make_tuple(cs("linear"),          Cast::linear),
  std::make_tuple(cs("localnorm"),       Cast::localnorm),
  std::make_tuple(cs("logsigmoid"),      Cast::logsigmoid),
  std::make_tuple(cs("logsoftmax"),      Cast::logsoftmax),
  std::make_tuple(cs("lppool1d"),        Cast::lppool1d),
  std::make_tuple(cs("lppool2d"),        Cast::lppool2d),
  std::make_tuple(cs("lstm"),            Cast::lstm),
  std::make_tuple(cs("maxpool1d"),       Cast::maxpool1d),
  std::make_tuple(cs("maxpool2d"),       Cast::maxpool2d),
  std::make_tuple(cs("maxpool3d"),       Cast::maxpool3d),
  std::make_tuple(cs("normalize"),       Cast::normalize),
  std::make_tuple(cs("pad"),             Cast::pad),
  std::make_tuple(cs("pad1d"),           Cast::pad1d),
  std::make_tuple(cs("pad2d"),           Cast::pad2d),
  std::make_tuple(cs("pad3d"),           Cast::pad3d),
  std::make_tuple(cs("pairwise"),        Cast::pairwise),
  std::make_tuple(cs("prelu"),           Cast::prelu),
  std::make_tuple(cs("reflect1d"),       Cast::reflect1d),
  std::make_tuple(cs("reflect2d"),       Cast::reflect2d),
  std::make_tuple(cs("relu"),            Cast::relu),
  std::make_tuple(cs("relu6"),           Cast::relu6),
  std::make_tuple(cs("replicate1d"),     Cast::replicate1d),
  std::make_tuple(cs("replicate2d"),     Cast::replicate2d),
  std::make_tuple(cs("replicate3d"),     Cast::replicate3d),
  std::make_tuple(cs("reshape"),         Cast::reshape),
  std::make_tuple(cs("rnn"),             Cast::rnn),
  std::make_tuple(cs("rrelu"),           Cast::rrelu),
  std::make_tuple(cs("selu"),            Cast::selu),
  std::make_tuple(cs("seqjoin"),         Cast::seqjoin),
  std::make_tuple(cs("seqnest"),         Cast::seqnest),
  std::make_tuple(cs("sequential"),      Cast::sequential),
  std::make_tuple(cs("sigmoid"),         Cast::sigmoid),
  std::make_tuple(cs("similar"),         Cast::similar),
  std::make_tuple(cs("softmax"),         Cast::softmax),
  std::make_tuple(cs("softmax2d"),       Cast::softmax2d),
  std::make_tuple(cs("softmin"),         Cast::softmin),
  std::make_tuple(cs("softplus"),        Cast::softplus),
  std::make_tuple(cs("softshrink"),      Cast::softshrink),
  std::make_tuple(cs("softsign"),        Cast::softsign),
  std::make_tuple(cs("squeeze"),         Cast::squeeze),
  std::make_tuple(cs("tanh"),            Cast::tanh),
  std::make_tuple(cs("tanhshrink"),      Cast::tanhshrink),
  std::make_tuple(cs("threshold"),       Cast::threshold),
  std::make_tuple(cs("unfold"),          Cast::unfold),
  std::make_tuple(cs("unsqueeze"),       Cast::unsqueeze),
  std::make_tuple(cs("zeropad2d"),       Cast::zeropad2d)
 }};

 std::array<std::tuple<S,Setting>,66> mset = {{        // module option sym -> enum
  std::make_tuple(cs("affine"),     Setting::affine),
  std::make_tuple(cs("alpha"),      Setting::alpha),
  std::make_tuple(cs("batchfirst"), Setting::batchfirst),
  std::make_tuple(cs("beta"),       Setting::beta),
  std::make_tuple(cs("bi"),         Setting::bi),
  std::make_tuple(cs("bias"),       Setting::bias),
  std::make_tuple(cs("ceiling"),    Setting::ceiling),
  std::make_tuple(cs("channels"),   Setting::channels),
  std::make_tuple(cs("cols"),       Setting::cols),
  std::make_tuple(cs("countpad"),   Setting::countpad),
  std::make_tuple(cs("dilate"),     Setting::dilate),
  std::make_tuple(cs("divisor"),    Setting::divisor),
  std::make_tuple(cs("dim"),        Setting::dim),
  std::make_tuple(cs("dropout"),    Setting::dropout),
  std::make_tuple(cs("end"),        Setting::end),
  std::make_tuple(cs("eps"),        Setting::eps),
  std::make_tuple(cs("fn"),         Setting::fn),
  std::make_tuple(cs("freeze"),     Setting::freeze),
  std::make_tuple(cs("groups"),     Setting::groups),
  std::make_tuple(cs("heads"),      Setting::heads),
  std::make_tuple(cs("hidden"),     Setting::hidden),
  std::make_tuple(cs("in"),         Setting::in),
  std::make_tuple(cs("in1"),        Setting::in1),
  std::make_tuple(cs("in2"),        Setting::in2),
  std::make_tuple(cs("indices"),    Setting::indices),
  std::make_tuple(cs("init"),       Setting::init),
  std::make_tuple(cs("inplace"),    Setting::inplace),
  std::make_tuple(cs("k"),          Setting::k),
  std::make_tuple(cs("kdim"),       Setting::kdim),
  std::make_tuple(cs("keepdim"),    Setting::keepdim),
  std::make_tuple(cs("kvbias"),     Setting::kvbias),
  std::make_tuple(cs("kvzeros"),    Setting::kvzeros),
  std::make_tuple(cs("lambda"),     Setting::lambda),
  std::make_tuple(cs("lastoffset"), Setting::lastoffset),
  std::make_tuple(cs("layers"),     Setting::layers),
  std::make_tuple(cs("lower"),      Setting::lower),
  std::make_tuple(cs("max"),        Setting::max),
  std::make_tuple(cs("maxnorm"),    Setting::maxnorm),
  std::make_tuple(cs("min"),        Setting::min),
  std::make_tuple(cs("mode"),       Setting::mode),
  std::make_tuple(cs("momentum"),   Setting::momentum),
  std::make_tuple(cs("out"),        Setting::out),
  std::make_tuple(cs("outpad"),     Setting::outpad),
  std::make_tuple(cs("outsize"),    Setting::outsize),
  std::make_tuple(cs("p"),          Setting::p),
  std::make_tuple(cs("pad"),        Setting::pad),
  std::make_tuple(cs("padindex"),   Setting::padindex),
  std::make_tuple(cs("padmode"),    Setting::padmode),
  std::make_tuple(cs("ratio"),      Setting::ratio),
  std::make_tuple(cs("rows"),       Setting::rows),
  std::make_tuple(cs("scale"),      Setting::scale),
  std::make_tuple(cs("size"),       Setting::size),
  std::make_tuple(cs("shape"),      Setting::shape),
  std::make_tuple(cs("slope"),      Setting::slope),
  std::make_tuple(cs("sparse"),     Setting::sparse),
  std::make_tuple(cs("start"),      Setting::start),
  std::make_tuple(cs("stride"),     Setting::stride),
  std::make_tuple(cs("threshold"),  Setting::threshold),
  std::make_tuple(cs("track"),      Setting::track),
  std::make_tuple(cs("train"),      Setting::train),
  std::make_tuple(cs("transpose"),  Setting::transpose),
  std::make_tuple(cs("type"),       Setting::type),
  std::make_tuple(cs("upper"),      Setting::upper),
  std::make_tuple(cs("value"),      Setting::value),
  std::make_tuple(cs("vdim"),       Setting::vdim),
  std::make_tuple(cs("weight"),     Setting::weight)
 }};

 std::array<std::tuple<S,State>,6> state = {{        //module state dictionary keys: map symbol -> enum
  std::make_tuple(cs("depth"),   State::depth),
  std::make_tuple(cs("module"),  State::module),
  std::make_tuple(cs("name"),    State::name),
  std::make_tuple(cs("options"), State::options),
  std::make_tuple(cs("parms"),   State::parms),
  std::make_tuple(cs("buffers"), State::buffers)
 }};

 std::array<std::tuple<S,Cast>,20> loss = {{             // loss: map symbol -> enum
  std::make_tuple(cs("bce"),          Cast::bce),
  std::make_tuple(cs("bcelogits"),    Cast::bcelogits),
  std::make_tuple(cs("ce"),           Cast::ce),
  std::make_tuple(cs("cosineloss"),   Cast::cosineloss),
  std::make_tuple(cs("ctc"),          Cast::ctc),
  std::make_tuple(cs("hinge"),        Cast::hinge),
  std::make_tuple(cs("kl"),           Cast::kl),
  std::make_tuple(cs("l1"),           Cast::l1),
  std::make_tuple(cs("margin"),       Cast::margin),
  std::make_tuple(cs("mse"),          Cast::mse),
  std::make_tuple(cs("multilabel"),   Cast::multilabel),
  std::make_tuple(cs("multimargin"),  Cast::multimargin),
  std::make_tuple(cs("multisoft"),    Cast::multisoft),
  std::make_tuple(cs("nll"),          Cast::nll),
  std::make_tuple(cs("pairwise"),     Cast::pairwise),
  std::make_tuple(cs("poissonloss"),  Cast::poissonloss),
  std::make_tuple(cs("similar"),      Cast::similar),
  std::make_tuple(cs("smoothl1"),     Cast::smoothl1),
  std::make_tuple(cs("softmargin"),   Cast::softmargin),
  std::make_tuple(cs("triplet"),      Cast::triplet)
 }};

 std::array<std::tuple<S,Setting>,11> lset = {{          // loss option sym -> enum
  std::make_tuple(cs("blank"),     Setting::blank),
  std::make_tuple(cs("eps"),       Setting::eps),
  std::make_tuple(cs("full"),      Setting::full),
  std::make_tuple(cs("ignore"),    Setting::ignore),
  std::make_tuple(cs("log"),       Setting::log),
  std::make_tuple(cs("margin"),    Setting::margin),
  std::make_tuple(cs("p"),         Setting::p),
  std::make_tuple(cs("reduce"),    Setting::reduce),
  std::make_tuple(cs("swap"),      Setting::swap),
  std::make_tuple(cs("weight"),    Setting::weight),
  std::make_tuple(cs("zeroinf"),   Setting::zeroinf)
 }};

 std::array<std::tuple<S,Cast,double>,5> opt = {{        //optimizer: map symbol -> enum, default learning rate PATCH
  std::make_tuple(cs("adagrad"), Cast::adagrad, 0.010),
  std::make_tuple(cs("adam"),    Cast::adam,    0.001),
  std::make_tuple(cs("lbfgs"),   Cast::lbfgs,   1.000),
  std::make_tuple(cs("rmsprop"), Cast::rmsprop, 0.010),
  std::make_tuple(cs("sgd"),     Cast::sgd,     0.010)
 }};

 std::array<std::tuple<S,Setting>,17> oset = {{         //optimizer setting: map symbol -> enum
  std::make_tuple(cs("lr"),         Setting::lr),
  std::make_tuple(cs("lrdecay"),    Setting::lrdecay),
  std::make_tuple(cs("decay"),      Setting::decay),
  std::make_tuple(cs("beta1"),      Setting::beta1),
  std::make_tuple(cs("beta2"),      Setting::beta2),
  std::make_tuple(cs("eps"),        Setting::eps),
  std::make_tuple(cs("amsgrad"),    Setting::amsgrad),
  std::make_tuple(cs("iter"),       Setting::iter),
  std::make_tuple(cs("eval"),       Setting::eval),
  std::make_tuple(cs("gradtol"),    Setting::gradtol),
  std::make_tuple(cs("changetol"),  Setting::changetol),
  std::make_tuple(cs("history"),    Setting::history),
  std::make_tuple(cs("alpha"),      Setting::alpha),
  std::make_tuple(cs("momentum"),   Setting::momentum),
  std::make_tuple(cs("centered"),   Setting::centered),
  std::make_tuple(cs("dampening"),  Setting::dampening),
  std::make_tuple(cs("nesterov"),   Setting::nesterov)
 }};

 std::array<std::tuple<S,Attr>,21> attr = {{            //attributes: map symbol -> enum
  std::make_tuple(cs("coalesced"),   Attr::coalesced),
  std::make_tuple(cs("contiguous"),  Attr::contiguous),
  std::make_tuple(cs("data"),        Attr::data),
  std::make_tuple(cs("device"),      Attr::device),
  std::make_tuple(cs("dim"),         Attr::dim),
  std::make_tuple(cs("dtype"),       Attr::dtype),
  std::make_tuple(cs("elementsize"), Attr::elementsize),
  std::make_tuple(cs("gradfn"),      Attr::gradfn),
  std::make_tuple(cs("gradient"),    Attr::gradient),
  std::make_tuple(cs("layout"),      Attr::layout),
  std::make_tuple(cs("leaf"),        Attr::leaf),
  std::make_tuple(cs("numel"),       Attr::numel),
  std::make_tuple(cs("offset"),      Attr::offset),
  std::make_tuple(cs("pinned"),      Attr::pinned),
  std::make_tuple(cs("ptr"),         Attr::ptr),
  std::make_tuple(cs("ref"),         Attr::ref),
  std::make_tuple(cs("size"),        Attr::size),
  std::make_tuple(cs("sparsedim"),   Attr::sparsedim),
  std::make_tuple(cs("storage"),     Attr::storage),
  std::make_tuple(cs("stride"),      Attr::stride),
  std::make_tuple(cs("weakref"),     Attr::weakref)
 }};

 std::array<std::tuple<S,bool,bool>,4> backsym = {{     //map sym to booleans for retain_graph & create_graph
  std::make_tuple(cs("free"),       false, false),
  std::make_tuple(cs("retain"),     true,  false),
  std::make_tuple(cs("create"),     true,  true),
  std::make_tuple(cs("createfree"), false, true)
 }};

 std::array<std::tuple<S,Metric>,4> metric = {{
  std::make_tuple(cs("loss"),       Metric::loss),
  std::make_tuple(cs("accuracy"),   Metric::accuracy),
  std::make_tuple(cs("max"),        Metric::max),
  std::make_tuple(cs("out"),        Metric::out)
 }};

 // array must match order of Enum, so enum can be used as index
 std::array<std::tuple<S,Enum>,30> enums = {{
  std::make_tuple(cs("area"),            Enum::area),
  std::make_tuple(cs("batchmean"),       Enum::batchmean),
  std::make_tuple(cs("bicubic"),         Enum::bicubic),
  std::make_tuple(cs("bilinear"),        Enum::bilinear),
  std::make_tuple(cs("border"),          Enum::border),
  std::make_tuple(cs("circular"),        Enum::circular),
  std::make_tuple(cs("constant"),        Enum::constant),
  std::make_tuple(cs("conv1d"),          Enum::conv1d),
  std::make_tuple(cs("conv2d"),          Enum::conv2d),
  std::make_tuple(cs("conv3d"),          Enum::conv3d),
  std::make_tuple(cs("convtranspose1d"), Enum::convtranspose1d),
  std::make_tuple(cs("convtranspose2d"), Enum::convtranspose2d),
  std::make_tuple(cs("convtranspose3d"), Enum::convtranspose3d),
  std::make_tuple(cs("fanin"),           Enum::fanin),
  std::make_tuple(cs("fanout"),          Enum::fanout),
  std::make_tuple(cs("leakyrelu"),       Enum::leakyrelu),
  std::make_tuple(cs("linear"),          Enum::linear),
  std::make_tuple(cs("max"),             Enum::max),
  std::make_tuple(cs("mean"),            Enum::mean),
  std::make_tuple(cs("nearest"),         Enum::nearest),
  std::make_tuple(cs("none"),            Enum::none),
  std::make_tuple(cs("reflect"),         Enum::reflect),
  std::make_tuple(cs("reflection"),      Enum::reflection),
  std::make_tuple(cs("relu"),            Enum::relu),
  std::make_tuple(cs("replicate"),       Enum::replicate),
  std::make_tuple(cs("sigmoid"),         Enum::sigmoid),
  std::make_tuple(cs("sum"),             Enum::sum),
  std::make_tuple(cs("tanh"),            Enum::tanh),
  std::make_tuple(cs("trilinear"),       Enum::trilinear),
  std::make_tuple(cs("zeros"),           Enum::zeros)
 }};
} Env;

Env& env();
std::unordered_set<J>& pointer();

// pytorch enumerations defined as variants (https://github.com/pytorch/pytorch/issues/15149)
// use structure to work with c10::vist to handle different enumeration tokens (structures)

struct Esym {
 S operator()(const torch::enumtype::kArea&)             const { return std::get<0>(env().enums[(size_t)Enum::area]);}
 S operator()(const torch::enumtype::kBatchMean&)        const { return std::get<0>(env().enums[(size_t)Enum::batchmean]);}
 S operator()(const torch::enumtype::kBicubic&)          const { return std::get<0>(env().enums[(size_t)Enum::bicubic]);}
 S operator()(const torch::enumtype::kBilinear&)         const { return std::get<0>(env().enums[(size_t)Enum::bilinear]);}
 S operator()(const torch::enumtype::kBorder&)           const { return std::get<0>(env().enums[(size_t)Enum::border]);}
 S operator()(const torch::enumtype::kCircular&)         const { return std::get<0>(env().enums[(size_t)Enum::circular]);}
 S operator()(const torch::enumtype::kConstant&)         const { return std::get<0>(env().enums[(size_t)Enum::constant]);}
 S operator()(const torch::enumtype::kConv1D&)           const { return std::get<0>(env().enums[(size_t)Enum::conv1d]);}
 S operator()(const torch::enumtype::kConv2D&)           const { return std::get<0>(env().enums[(size_t)Enum::conv2d]);}
 S operator()(const torch::enumtype::kConv3D&)           const { return std::get<0>(env().enums[(size_t)Enum::conv3d]);}
 S operator()(const torch::enumtype::kConvTranspose1D&)  const { return std::get<0>(env().enums[(size_t)Enum::convtranspose1d]);}
 S operator()(const torch::enumtype::kConvTranspose2D&)  const { return std::get<0>(env().enums[(size_t)Enum::convtranspose2d]);}
 S operator()(const torch::enumtype::kConvTranspose3D&)  const { return std::get<0>(env().enums[(size_t)Enum::convtranspose3d]);}
 S operator()(const torch::enumtype::kFanIn&)            const { return std::get<0>(env().enums[(size_t)Enum::fanin]);}
 S operator()(const torch::enumtype::kFanOut&)           const { return std::get<0>(env().enums[(size_t)Enum::fanout]);}
 S operator()(const torch::enumtype::kLeakyReLU&)        const { return std::get<0>(env().enums[(size_t)Enum::leakyrelu]);}
 S operator()(const torch::enumtype::kLinear&)           const { return std::get<0>(env().enums[(size_t)Enum::linear]);}
 S operator()(const torch::enumtype::kMax&)              const { return std::get<0>(env().enums[(size_t)Enum::max]);}
 S operator()(const torch::enumtype::kMean&)             const { return std::get<0>(env().enums[(size_t)Enum::mean]);}
 S operator()(const torch::enumtype::kNearest&)          const { return std::get<0>(env().enums[(size_t)Enum::nearest]);}
 S operator()(const torch::enumtype::kNone&)             const { return std::get<0>(env().enums[(size_t)Enum::none]);}
 S operator()(const torch::enumtype::kReflect&)          const { return std::get<0>(env().enums[(size_t)Enum::reflect]);}
 S operator()(const torch::enumtype::kReflection&)       const { return std::get<0>(env().enums[(size_t)Enum::reflection]);}
 S operator()(const torch::enumtype::kReLU&)             const { return std::get<0>(env().enums[(size_t)Enum::relu]);}
 S operator()(const torch::enumtype::kReplicate&)        const { return std::get<0>(env().enums[(size_t)Enum::replicate]);}
 S operator()(const torch::enumtype::kSigmoid&)          const { return std::get<0>(env().enums[(size_t)Enum::sigmoid]);}
 S operator()(const torch::enumtype::kSum&)              const { return std::get<0>(env().enums[(size_t)Enum::sum]);}
 S operator()(const torch::enumtype::kTanh&)             const { return std::get<0>(env().enums[(size_t)Enum::tanh]);}
 S operator()(const torch::enumtype::kTrilinear&)        const { return std::get<0>(env().enums[(size_t)Enum::trilinear]);}
 S operator()(const torch::enumtype::kZeros&)            const { return std::get<0>(env().enums[(size_t)Enum::zeros]);}
};

Esym& esym();
#define ESYM(v) c10::visit(esym(), v)
