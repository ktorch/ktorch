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
#include "kmodel.h"
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

#define TORCH_ERROR(...)                                                      \
  do {                                                                        \
    C10_EXPAND_MSVC_WORKAROUND(TORCH_CHECK(false, ::c10::str(__VA_ARGS__)));  \
  } while (false)

#define KFN(f) reinterpret_cast<void *>(f)
#define KERR(e) krr((S)e)
#define KDICT xD(ktn(KS,0),ktn(0,0))

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
using Device=torch::Device;
using DeviceType=torch::DeviceType;
using Storage=torch::Storage;
using Tensor=torch::Tensor;
using Scalar=torch::Scalar;
using TensorVector=std::vector<Tensor>;
using TensorDeque=std::deque<Tensor>;
using LongVector=std::vector<int64_t>;
using DoubleVector=std::vector<double>;
using IntArrayRef=torch::IntArrayRef;
using DoubleArrayRef=torch::ArrayRef<double>;
template<size_t D,typename T=int64_t> using ExpandingArray=torch::ExpandingArray<D,T>;
template<size_t D,typename T=double>  using Exdouble=torch::ExpandingArray<D,T>;
template<size_t D,typename T=int64_t> using Exoptional=torch::ExpandingArrayWithOptionalElem<D,T>;
using Dtype=torch::Dtype;
using TypeMeta=caffe2::TypeMeta;
using ScalarType=torch::ScalarType;
using TensorOptions=torch::TensorOptions;
using TensorList=torch::TensorList;  // ArrayRef<Tensor>
using TensorDict=torch::OrderedDict<std::string, torch::Tensor>;

// shorter names for commonly used container modules defined by pytorch & created in knn.h
using Module=torch::nn::Module;
using Moduleptr=std::shared_ptr<Module>;
using Modules=std::stack<Moduleptr>;
using Modulemap=torch::OrderedDict<std::string, std::shared_ptr<Module>>;
using Modulepairs=std::vector<std::pair<std::string, std::shared_ptr<Module>>>;
using AnyModule=torch::nn::AnyModule;
class SeqNest;
class SeqJoin;

using Tuple         = std::tuple<Tensor,Tensor>;
using Tensors       = std::array<Tensor,3>;
using TensorVectors = std::array<TensorVector,2>;
using Input         = c10::variant<TensorVector,TensorVectors,TensorDict>; 
using Output        = c10::variant<Tensor,Tuple,Tensors,TensorVector>;

using Optimizer=torch::optim::Optimizer;
using Optptr=std::shared_ptr<Optimizer>;

typedef struct {
 Ktype a = 0;  // type: 1-dict, 2-list of pairs, 3-general list, 4-sym list
 Ktype t = 0;  // type of value in last pair processed
 H i = 0;  // next pair to process
 H n = 0;  // count of pairs
 S k = 0;  // name of current name,value pair
 K x = 0;  // k value with dict/pairs/list
 union {
  bool   b;  // boolean value from current pair
  J      j;  // long value
  double f;  // double value
  S      s;  // symbol value
  K      v;  // value (isn't sym or numeric scalar)
 };
} Pairs;

enum class Class:short {
 undefined=0,
 tensor,
 vector,
 dict,
 module,
 layer,
 sequential,
 loss,
 optimizer,
 model
};

enum class Cast:short {
 undefined=0, 
 tensor,      model,                                     // basic structures
 base,        moduledict,   modulelist,      parmdict,   // container modules
 sequential,  seqnest,      seqjoin,
 adaptavg1d,  adaptavg2d,   adaptavg3d,      adaptmax1d,      adaptmax2d,      adaptmax3d,   // modules
 adrop,       attention,    avgpool1d,       avgpool2d,       avgpool3d,       batchnorm1d,
 batchnorm2d, batchnorm3d,  bilinear,        cat,             celu,            conv1d,     
 conv2d,      conv3d,       convtranspose1d, convtranspose2d, convtranspose3d, crossmap2d, 
 decoder,     decoderlayer, drop,            drop2d,          drop3d,          elu,        
 embed,       embedbag,     encoder,         encoderlayer,    expand,          fadrop,     
 flatten,     fmaxpool2d,   fmaxpool3d,      fold,            fork,            gelu,       
 glu,         groupnorm,    gru,             gruout,          hardshrink,      hardtanh,   
 identity,    indexselect,  instancenorm1d,  instancenorm2d,  instancenorm3d,  interpolate,
 layernorm,   leakyrelu,    linear,          localnorm,       logsigmoid,      logsoftmax, 
 lppool1d,    lppool2d,     lstm,            lstmout,         maxpool1d,       maxpool2d,  
 maxpool3d,   mish,         mul,             nbeats,          normalize,       onehot,     
 pad,         pad1d,        pad2d,           pad3d,           prelu,           randomcrop, 
 randomflip,  recur,        reflect1d,       reflect2d,       relu,            relu6,      
 replicate1d, replicate2d,  replicate3d,     reshape,         residual,        rnn,        
 rnnout,      rrelu,        select,          selu,            sigmoid,         silu,       
 softmax,     softmax2d,    softmin,         softplus,        softshrink,      softsign,   
 squeeze,     tanh,         tanhshrink,      threshold,       transform,       transformer,
 unfold,      unsqueeze,    upsample,        zeropad2d,       zscore,                      
 pairwise,    similar,                                                                       // distance functions
 bce,         bcelogits,    ce,               cosineloss,     ctc,             hinge,        // loss fns
 huber,       kl,           l1,               margin,         mse,             multilabel,
 multimargin, multisoft,    nll,              poissonloss,    sce,             smoothl1,
 softmargin,  triplet,    
 adagrad,     adam,         adamw,            lbfgs,          rmsprop,         sgd           // optimizers
};

enum class Tensormode:char {   // tensor creation modes
 undefined,
 arange, complex, empty,   eye,   full,     linspace, logspace,
 ones,   rand,    randint, randn, randperm, range,    sparse, 
 zeros    
};

enum class Prob:char {  // probablility distributions
 cauchy, exponential, geometric, lognormal, normal, random, uniform
};

enum class Init:char {
 constant,   dirac,  eye,     kaimingnormal, kaiminguniform, normal, ones,
 orthogonal, sparse, uniform, xaviernormal,  xavieruniform,  zeros
};

enum class Setting:char {
 undefined,
 addbias,      addzero,    affine,    align,        alloptions, alpha,              amsgrad,       
 batchfirst,   benchmark,  beta,      beta1,        beta2,      bi,                 bias,          
 blank,        ceiling,    centered,  changetol,    channels,   classes,            cols,          
 complexfirst, countpad,   cuda,      cudadevices,  cudnn,      cudnndeterministic, cudnnversion,  
 dampening,    decay,      decoder,   decoderlayer, delta, detach,     deterministic,      dilate,        
 dim,          divisor,    dlayers,   dropout,      dtype,      elayers,            encoder,       
 encoderlayer, end,        eps,       eval,         fn,         freeze,             full,          
 gradtol,      groups,     heads,     hidden,       history,    ignore,             in,            
 in1,          in2,        ind,       indices,      init,       inplace,            interopthreads,
 iter,         k,          kdim,      keepdim,      kvbias,     kvzeros,            lambda,        
 lastoffset,   layernorm,  layers,    log,          lower,      lr,                 lrdecay,       
 magma,        margin,     max,       maxnorm,      mean, min,        mkl,                mode,          
 momentum,     nesterov,   norm,      openmp,       out,        outpad,             outsize,       
 p,            pad,        padflag,   padindex,  padmode,      ratio,      reduce,             rescale,       
 rows,         scale,      search,    shape,        size,       slope,              smoothing,
 sparse,       stackframe, start,     std, stride,       swap,       threads,            threshold,
 track,        train,      transpose, upper,        value,      vdim,               weight,
 zeroinf        
};

enum class State:char {
 buffers, depth, module, name, options, optlist, optimizer, parms, parmgroup, pointer, size
};

enum class Attr:char {
 undefined = 0,
 bytes,  densedim, dim, elements,  itemsize, nnz, numel, offset, // long scalars
 ptr, ref, sparsedim, sptr, sref, tensorcount, weakref,
 device, dtype, gradfn, gradient, layout, memory, result,       // symbol
 coalesced, contiguous, contiguous2d, contiguous3d, defined,    // boolean
 gradflag, leaf, pinned, sparseflag,
 size, stride,                                                  // long list
 data, storage                                                  // other: list,dict,..
};
 
enum class Metric: char {
 loss, accuracy, max, out
};

enum class Enum {  // enums to match pytorch variants
 undefined=-1,
 area,            batchmean,       bicubic,         bilinear,   border,   
 circular,        constant,        conv1d,          conv2d,     conv3d,   
 convtranspose1d, convtranspose2d, convtranspose3d, fanin,      fanout,   
 gelu,            leakyrelu,       linear,          max,        mean,
 mish,            nearest,         none,            reflect,    reflection,
 relu,            replicate,       same,            sigmoid,    silu,
 sum,             tanh,            trilinear,       valid,      zeros
};

struct TORCH_API Ktag {
 Class  a = Class::undefined;
 Cast   c = Cast::undefined;
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

struct TORCH_API Kdict : public Ktag {
 TensorDict d;
 Kdict(const TensorDict& x) : d(std::move(x)) {a=Class::dict; c=Cast::tensor;}
};

struct TORCH_API Kmodule : public Ktag {
 Kmodule(Class x,Cast y,Moduleptr p) : m(std::move(p)) {a=x; c=y;}
 Moduleptr m;
};

struct TORCH_API Kopt : public Ktag {
 Optptr o;       // shared ptr with optimizer
 Moduleptr m;    // single module or container holding all modules/tensors managed by optimizer
 Kopt(Cast x,const Optptr& y,const Moduleptr& m) : o(std::move(y)),m(std::move(m)) {a=Class::optimizer; c=x;}
};

struct TORCH_API Kmodel : public Ktag {
 Cast mc;             // type of module, typically a container module, e.g. Sequential
 Cast lc;             // type of loss fn
 Cast oc;             // type of optimizer
 Moduleptr m;         // generic pointer to top-level module, e.g. Sequential
 Moduleptr l;         // loss module
 Optptr o;            // shared ptr to optimizer
 Moduleptr om;        // single module or container holding all modules/tensors managed by optimizer
 TrainOptions train;  // training options
 EvalOptions eval;    // evaluation options
 Kmodel(Kmodule *x,Kmodule *y,Kopt *z) : mc(x->c),lc(y->c),oc(z->c),m(x->m),l(y->m),o(z->o),om(z->m) {
  a=Class::model; c=Cast::model;
 }
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
void print_tensor(std::ostream&,int64_t,const Tensor& t);
Enum emap(S);
S emap(Enum);

S statekey(State);
J statefind(State,K,bool r=false);
J statedepth(K x,J j=-1);
S statemodule(K x,J j=-1);
S statename(K x,J j=-1);
S stateoptimizer(K x,J j=-1);
K stateoptions(K x,J j=-1);
K stateoptlist(K x,J j=-1);
K stateparms(K x,J j=-1);
K statebuffers(K x,J j=-1);
J stategroup(K x,J j=-1);
K statesize(K x,J j=-1);
K stategroups(K);
K statecol(State,K,short t=nh);
void stateparms(S,Module&,K,bool);

S nullsym();
bool nullsym(S);
bool nullsym(K);

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
bool xdev(K,Device&);
bool xdev(K,J,Device&);

bool xint64(K,int64_t&);
bool xint64(K,J,int64_t&);
bool xint64(K,c10::optional<int64_t>&);
bool xint64(K,J,c10::optional<int64_t>&);
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
Tensor* xout(K);
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
Kmodule* xloss(K);
Kmodule* xloss(K,J);
Kopt* xoptim(K);
Kopt* xoptim(K,J);
Kmodel* xmodel(K);
Kmodel* xmodel(K,J);
TensorVector* xvec(K);
TensorVector* xvec(K,J);
TensorDict* xtensordict(K);
TensorDict* xtensordict(K,J);

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
Dtype stype(S);
S stype(Dtype);
S stype(c10::optional<Dtype>);
bool xtype(K,Dtype&);
bool xtype(K,J,Dtype&);
bool xtype(K,c10::optional<Dtype>&);
bool xtype(K,J,c10::optional<Dtype>&);
bool xtype(K,TypeMeta&);
bool xtype(K,J,TypeMeta&);
bool xopt(S,TensorOptions&);
bool xopt(K,TensorOptions&);
bool xopt(K,J,TensorOptions&);
bool xmode(K,S&,Tensormode&);
bool xmode(K,J,S&,Tensormode&);
S    modesym(Tensormode&);
bool xbacksym(K,bool&,bool&);
bool xbacksym(K,J,bool&,bool&);

bool xpairs(K,Pairs&);
bool xpairs(K,J,Pairs&);
bool xpair(Pairs&);
J xargc(K,J,Pairs&);
bool xnone(K,J);

S psym(const Pairs&);
Dtype ptype(const Pairs&);
void perr(const Pairs&,const char*);
bool pempty(const Pairs&);
bool pbool(const Pairs&);
J plong(const Pairs&);
double pdouble(const Pairs&);
void pnum(const Pairs&,Scalar&);
void psize(const Pairs&,IntArrayRef&,J n=-1);
void psize(const Pairs&,J,int64_t*);
void psize(const Pairs&,J,double*);
void pdoubles(const Pairs&,DoubleArrayRef&,J n=-1);
void pten(const Pairs&,Tensor&);

S& optdev(const Device&);
S& optdtype(const TypeMeta&);
S& optdtype(ScalarType);
S& optlayout(const torch::Layout&);
S& optmemory(const c10::optional<torch::MemoryFormat>&);
torch::MemoryFormat optmemory(S);
S& optgrad(const bool&);
S& optpin(const bool&);
K optkey();
K optval(const Tensor &t,K x,J i=-1);
K optval(const TensorOptions &o,K x,J i=-1);
K optmap(const Tensor&);
K optmap(const TensorOptions&);
std::string kstring(K);
std::string kstring(K,J);
K kshow(K);
K kcast(Ktype,K);
K kbool(K);
J kfind(K,const std::string&);
K klist(J,const int64_t*);
K klist(J,const double*);
K klist(J,const c10::optional<int64_t>*);
K kexpand(J,const int64_t*);
K kexpand(J,const double*);
K kexpand(J,const c10::optional<int64_t>*e);
#define KEX(x) kexpand(x.size(),(*x).data())  // k list from ExpandingArray
J xdv(K);
J xdv(K,J);
J dvd(K,J);
K dvv(K,J);

S objdevice(const Tensor&);
S objdevice(const TensorVector&,S);
J objnum(int64_t);
J objnum(double);
J objnum(const Tensor&);
J objnum(const TensorVector&);
J objnum(const c10::optional<TensorVector>&);
J objnum(const TensorDeque&);
J objnum(const Module&);
J objnum(Cast,const Optimizer&);
J objbytes(int64_t);
J objbytes(double);
J objbytes(const Tensor&);
J objbytes(const TensorVector&);
J objbytes(const c10::optional<TensorVector>&);
J objbytes(const TensorDeque&);
J objbytes(const Module&);
J objbytes(Cast,const Optimizer&);

bool kfree(K);
bool kfree(K,J);
void kfree(const std::vector<K>&);
void fn(K,const char*,void*,I);
void randomfn(K);
void mathfn(K);
K attr(K,Ktype,Attr);

// tensor & vector routines:
K kget(const Tensor&);
K kget(const LongVector&);
K kget(const DoubleVector&);
K kget(const TensorVector& v,K x=nullptr);
K kget(const TensorDict& d,K x=nullptr);
K kget(const TensorDeque&);
K kget(const Tuple&);
K kget(const Tensors&);
K kget(const Output&);
TensorVector tvec(const Tensors&);
K kout(const Output&);
Tensor kput(K);
Tensor kput(K,J);
TensorDict kputd(K);
TensorVector vec(K,bool b=false);
K kten(const Tensor&);
K kvec(const TensorVector&);
K kdict(const TensorDict&);
inline K kresult(bool p,const Tensor& t) {return p ? kten(t) : kget(t);}
K to(Kten*,  const TensorOptions&,bool,bool);
K to(Kdict*, const TensorOptions&,bool);
K to(Kvec*,  const TensorOptions&,bool);
void to(TensorVector&,const TensorOptions&,bool);
K ktenpair(bool,Tensor&,Tensor&);
K kten3(bool,Tensor&,Tensor&,Tensor&);
J tensorlong(const Tensor&,Attr);
S tensorsym(const Tensor&,Attr);
K tensorsize(const Tensor&,Attr);
K tensorattr(const Tensor&,Ktype,Attr);
K vectorattr(const TensorVector&,Ktype,Attr);
K dictattr(const TensorDict&,Ktype,Attr);
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
void setsafe(Tensor& t,int64_t,const IntArrayRef&,const IntArrayRef&);
void tensorfn(K);

// nn module & functional routines:
// distance module arg set/get (for use in loss functions)
torch::nn::CosineSimilarityOptions  similar(K,J,Cast);
torch::nn::PairwiseDistanceOptions pairwise(K,J,Cast);
K  similar(bool,const torch::nn::CosineSimilarityOptions&);
K pairwise(bool,const torch::nn::PairwiseDistanceOptions&);

K kmodule(Cast c,const Moduleptr& m,Class a=Class::module);
void to(Module&,const TensorOptions&,bool);
K to(Kmodule*,const TensorOptions&,bool);

const
c10::optional<std::string>& mname_(const Module&);
c10::optional<std::string>& mname_(Module&);
S mname(const Module&);
std::string mlabel(const Module&);
std::string mlabel(const Moduleptr&);
std::string mlabel(Kmodule*);
Cast mcast(const Module&);
Cast msym(S);
S msym(Cast);
S msym(const Module&);
K mget(bool,bool,const Module&);
Output mforward(Cast,Module&,const Tensor&);
Output mforward(Cast,Module&,const Tensor&,const Tensor&);
Output mforward(Cast,Module&,const Tensor&,const Tensor&,const Tensor&);
K modulehelp(Cast);
void nnfn(K);

// loss functions:
K kloss(Cast,const Moduleptr&);
Cast lmap(S);
S lmap(Cast);
Tensor lossfwd(Cast,Module&,const Tensor&,const Tensor&);
Tensor lossfwd(Cast,Module&,const Tensor&,const Tensor&,const Tensor&);
Tensor lossfwd(Cast,Module&,const Tensor&,const Tensor&,const Tensor&,const Tensor&);
K lossdict(bool,bool,Cast,const Module&);
K losshelp(Cast);
void lossfn(K);

// optimization functions:
size_t osize(const Optimizer&);
J buffersize(Attr,Cast,const Optimizer&);
K kopt(Cast,const Optptr&,const Moduleptr&);
K optstate(bool,bool,Kopt *);
K optstate(bool,bool,Kmodel *);
K opthelp(Cast);
void optstep(Cast,Optptr&);
void optstep(Kopt*);
void optstep(Kmodel*);
void optfn(K);

// model functions:
K modelstate(bool,bool,Kmodel*);
K mbackward(K);
void modelfn(K);

// global environment
typedef struct {
 I cuda;                  // number of CUDA devices
 S nullsym=cs("");        // internal representation of null symbol
 bool frame=false;        // if true, error message returns stack frame
 bool alloptions=true;    // if true, return all option settings, else only non-defaults
 bool complexfirst=true;  // if true, return complex tensor as (real;imag) instead of (real,'imag)

 std::vector<std::tuple<S,Device>> device;

 std::array<std::tuple<Ktype,TypeMeta>,8> ktype = {{           //k type -> torch type
  std::make_tuple(KE, torch::scalarTypeToTypeMeta(torch::kFloat)),
  std::make_tuple(KF, torch::scalarTypeToTypeMeta(torch::kDouble)),
  std::make_tuple(KJ, torch::scalarTypeToTypeMeta(torch::kLong)),
  std::make_tuple(KI, torch::scalarTypeToTypeMeta(torch::kInt)),
  std::make_tuple(KH, torch::scalarTypeToTypeMeta(torch::kShort)),
  std::make_tuple(KB, torch::scalarTypeToTypeMeta(torch::kBool)),
  std::make_tuple(KG, torch::scalarTypeToTypeMeta(torch::kByte)),
  std::make_tuple(KC, torch::scalarTypeToTypeMeta(torch::kChar))
 }};

 std::array<std::tuple<S,TypeMeta,Ktype>,12> dtype = {{       //sym -> torch type -> k type
  std::make_tuple(cs("float"),   torch::scalarTypeToTypeMeta(torch::kFloat),         KE),
  std::make_tuple(cs("double"),  torch::scalarTypeToTypeMeta(torch::kDouble),        KF),
  std::make_tuple(cs("half"),    torch::scalarTypeToTypeMeta(torch::kHalf),          KE),
  std::make_tuple(cs("bool"),    torch::scalarTypeToTypeMeta(torch::kBool),          KB),
  std::make_tuple(cs("byte"),    torch::scalarTypeToTypeMeta(torch::kByte),          KG),
  std::make_tuple(cs("char"),    torch::scalarTypeToTypeMeta(torch::kChar),          KC),
  std::make_tuple(cs("long"),    torch::scalarTypeToTypeMeta(torch::kLong),          KJ),
  std::make_tuple(cs("int"),     torch::scalarTypeToTypeMeta(torch::kInt),           KI),
  std::make_tuple(cs("short"),   torch::scalarTypeToTypeMeta(torch::kShort),         KH),
  std::make_tuple(cs("chalf"),   torch::scalarTypeToTypeMeta(torch::kComplexHalf),   KH),
  std::make_tuple(cs("cfloat"),  torch::scalarTypeToTypeMeta(torch::kComplexFloat),  KH),
  std::make_tuple(cs("cdouble"), torch::scalarTypeToTypeMeta(torch::kComplexDouble), KH)
 }};

 std::array<std::tuple<S,torch::Layout>,2> layout = {{
  std::make_tuple(cs("strided"),torch::kStrided),          
  std::make_tuple(cs("sparse"), torch::kSparse)
 }};

 std::array<std::tuple<S,bool>,2> gradient = {{
  std::make_tuple(cs("grad"),   true),          
  std::make_tuple(cs("nograd"), false)
 }};

 std::array<std::tuple<S,bool>,2> pin = {{
  std::make_tuple(cs("pinned"),   true),          
  std::make_tuple(cs("unpinned"), false)
 }};

 std::array<std::tuple<S,torch::MemoryFormat>,4> memory = {{
  std::make_tuple(cs("contiguous"), torch::MemoryFormat::Contiguous),          
  std::make_tuple(cs("preserve"),   torch::MemoryFormat::Preserve),          
  std::make_tuple(cs("channel2d"),  torch::MemoryFormat::ChannelsLast),
  std::make_tuple(cs("channel3d"),  torch::MemoryFormat::ChannelsLast3d)
 }};

 std::array<std::tuple<S,Class>,7> kclass = {{         //higher level object names
  std::make_tuple(cs("tensor"),    Class::tensor),          
  std::make_tuple(cs("vector"),    Class::vector),
  std::make_tuple(cs("dict"),      Class::dict),
  std::make_tuple(cs("module"),    Class::module),
  std::make_tuple(cs("loss"),      Class::loss),
  std::make_tuple(cs("optimizer"), Class::optimizer),
  std::make_tuple(cs("model"),     Class::model)
 }};

 std::array<std::tuple<S,Class>,3> model = {{
  std::make_tuple(cs("module"), Class::module),
  std::make_tuple(cs("loss"),   Class::loss),
  std::make_tuple(cs("opt"),    Class::optimizer),
 }};

 std::array<std::tuple<S,Tensormode>,15> tensormode = {{    //tensor creation mode: map symbol -> enum
  std::make_tuple(cs("arange"),   Tensormode::arange),
  std::make_tuple(cs("complex"),  Tensormode::complex),
  std::make_tuple(cs("empty"),    Tensormode::empty),
  std::make_tuple(cs("eye"),      Tensormode::eye),
  std::make_tuple(cs("full"),     Tensormode::full),
  std::make_tuple(cs("linspace"), Tensormode::linspace),
  std::make_tuple(cs("logspace"), Tensormode::logspace),
  std::make_tuple(cs("ones"),     Tensormode::ones),
  std::make_tuple(cs("randint"),  Tensormode::randint),
  std::make_tuple(cs("randn"),    Tensormode::randn),
  std::make_tuple(cs("randperm"), Tensormode::randperm),
  std::make_tuple(cs("rand"),     Tensormode::rand),
  std::make_tuple(cs("range"),    Tensormode::range),
  std::make_tuple(cs("sparse"),   Tensormode::sparse),
  std::make_tuple(cs("zeros"),    Tensormode::zeros)
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

std::array<std::tuple<S,Init>,13> init = {{   //initialization methods
  std::make_tuple(cs("constant"),        Init::constant),
  std::make_tuple(cs("dirac"),           Init::dirac),
  std::make_tuple(cs("eye"),             Init::eye),
  std::make_tuple(cs("kaimingnormal"),   Init::kaimingnormal),
  std::make_tuple(cs("kaiminguniform"),  Init::kaiminguniform),
  std::make_tuple(cs("normal"),          Init::normal),
  std::make_tuple(cs("ones"),            Init::ones),
  std::make_tuple(cs("orthogonal"),      Init::orthogonal),
  std::make_tuple(cs("sparse"),          Init::sparse),
  std::make_tuple(cs("uniform"),         Init::uniform),
  std::make_tuple(cs("xaviernormal"),    Init::xaviernormal),
  std::make_tuple(cs("xavieruniform"),   Init::xavieruniform),
  std::make_tuple(cs("zeros"),           Init::zeros),
 }};

 std::array<std::tuple<S,Cast,size_t,std::string>,122> module = {{      // module sym -> enum, type id, pytorch name
  std::make_tuple(cs("adaptavg1d"),       Cast::adaptavg1d,      typeid(torch::nn::AdaptiveAvgPool1dImpl).hash_code(),   "torch.nn.AdaptiveAvgPool1d"),
  std::make_tuple(cs("adaptavg2d"),       Cast::adaptavg2d,      typeid(torch::nn::AdaptiveAvgPool2dImpl).hash_code(),   "torch.nn.AdaptiveAvgPool2d"),
  std::make_tuple(cs("adaptavg3d"),       Cast::adaptavg3d,      typeid(torch::nn::AdaptiveAvgPool3dImpl).hash_code(),   "torch.nn.AdaptiveAvgPool3d"),
  std::make_tuple(cs("adaptmax1d"),       Cast::adaptmax1d,      typeid(torch::nn::AdaptiveMaxPool1dImpl).hash_code(),   "torch.nn.AdaptiveMaxPool1d"),
  std::make_tuple(cs("adaptmax2d"),       Cast::adaptmax2d,      typeid(torch::nn::AdaptiveMaxPool2dImpl).hash_code(),   "torch.nn.AdaptiveMaxPool2d"),
  std::make_tuple(cs("adaptmax3d"),       Cast::adaptmax3d,      typeid(torch::nn::AdaptiveMaxPool3dImpl).hash_code(),   "torch.nn.AdaptiveMaxPool3d"),
  std::make_tuple(cs("adrop"),            Cast::adrop,           typeid(torch::nn::AlphaDropoutImpl).hash_code(),        "torch.nn.AlphaDropout"),
  std::make_tuple(cs("attention"),        Cast::attention,       typeid(torch::nn::MultiheadAttentionImpl).hash_code(),  "torch.nn.MultiheadAttention"),
  std::make_tuple(cs("avgpool1d"),        Cast::avgpool1d,       typeid(torch::nn::AvgPool1dImpl).hash_code(),           "torch.nn.AvgPool1d"),
  std::make_tuple(cs("avgpool2d"),        Cast::avgpool2d,       typeid(torch::nn::AvgPool2dImpl).hash_code(),           "torch.nn.AvgPool2d"),
  std::make_tuple(cs("avgpool3d"),        Cast::avgpool3d,       typeid(torch::nn::AvgPool3dImpl).hash_code(),           "torch.nn.AvgPool3d"),
  std::make_tuple(cs("base"),             Cast::base,            typeid(BaseModuleImpl).hash_code(),                     "torch.nn.Module"),
  std::make_tuple(cs("batchnorm1d"),      Cast::batchnorm1d,     typeid(torch::nn::BatchNorm1dImpl).hash_code(),         "torch.nn.BatchNorm1d"),
  std::make_tuple(cs("batchnorm2d"),      Cast::batchnorm2d,     typeid(torch::nn::BatchNorm2dImpl).hash_code(),         "torch.nn.BatchNorm2d"),
  std::make_tuple(cs("batchnorm3d"),      Cast::batchnorm3d,     typeid(torch::nn::BatchNorm3dImpl).hash_code(),         "torch.nn.BatchNorm3d"),
  std::make_tuple(cs("bilinear"),         Cast::bilinear,        typeid(torch::nn::BilinearImpl).hash_code(),            "torch.nn.Bilinear"),
  std::make_tuple(cs("cat"),              Cast::cat,             typeid(CatImpl).hash_code(),                            "torch.cat"),
  std::make_tuple(cs("celu"),             Cast::celu,            typeid(torch::nn::CELUImpl).hash_code(),                "torch.nn.CELU"),
  std::make_tuple(cs("conv1d"),           Cast::conv1d,          typeid(torch::nn::Conv1dImpl).hash_code(),              "torch.nn.Conv1d"),
  std::make_tuple(cs("conv2d"),           Cast::conv2d,          typeid(torch::nn::Conv2dImpl).hash_code(),              "torch.nn.Conv2d"),
  std::make_tuple(cs("conv3d"),           Cast::conv3d,          typeid(torch::nn::Conv3dImpl).hash_code(),              "torch.nn.Conv3d"),
  std::make_tuple(cs("convtranspose1d"),  Cast::convtranspose1d, typeid(torch::nn::ConvTranspose1dImpl).hash_code(),     "torch.nn.ConvTranspose1d"),
  std::make_tuple(cs("convtranspose2d"),  Cast::convtranspose2d, typeid(torch::nn::ConvTranspose2dImpl).hash_code(),     "torch.nn.ConvTranspose2d"),
  std::make_tuple(cs("convtranspose3d"),  Cast::convtranspose3d, typeid(torch::nn::ConvTranspose3dImpl).hash_code(),     "torch.nn.ConvTranspose3d"),
  std::make_tuple(cs("crossmap2d"),       Cast::crossmap2d,      typeid(torch::nn::CrossMapLRN2dImpl).hash_code(),       "torch.nn.CrossMapLRN2d"),
  std::make_tuple(cs("decoder"),          Cast::decoder,         typeid(torch::nn::TransformerDecoderImpl).hash_code(),  "torch.nn.TransformerDecoder"),
  std::make_tuple(cs("decoderlayer"),     Cast::decoderlayer,    typeid(torch::nn::TransformerDecoderLayerImpl).hash_code(), "torch.nn.TransformerDecoderLayer"),
  std::make_tuple(cs("drop"),             Cast::drop,            typeid(torch::nn::DropoutImpl).hash_code(),             "torch.nn.Dropout"),
  std::make_tuple(cs("drop2d"),           Cast::drop2d,          typeid(torch::nn::Dropout2dImpl).hash_code(),           "torch.nn.Dropout2d"),
  std::make_tuple(cs("drop3d"),           Cast::drop3d,          typeid(torch::nn::Dropout3dImpl).hash_code(),           "torch.nn.Dropout3d"),
  std::make_tuple(cs("elu"),              Cast::elu,             typeid(torch::nn::ELUImpl).hash_code(),                 "torch.nn.ELU"),
  std::make_tuple(cs("embed"),            Cast::embed,           typeid(torch::nn::EmbeddingImpl).hash_code(),           "torch.nn.Embedding"),
  std::make_tuple(cs("embedbag"),         Cast::embedbag,        typeid(torch::nn::EmbeddingBagImpl).hash_code(),        "torch.nn.EmbeddingBag"),
  std::make_tuple(cs("encoder"),          Cast::encoder,         typeid(torch::nn::TransformerEncoderImpl).hash_code(),  "torch.nn.TransformerEncoder"),
  std::make_tuple(cs("encoderlayer"),     Cast::encoderlayer,    typeid(torch::nn::TransformerEncoderLayerImpl).hash_code(), "torch.nn.TransformerEncoderLayer"),
  std::make_tuple(cs("expand"),           Cast::expand,          typeid(ExpandImpl).hash_code(),                         "torch.Tensor.expand"),
  std::make_tuple(cs("fadrop"),           Cast::fadrop,          typeid(torch::nn::FeatureAlphaDropoutImpl).hash_code(), "torch.nn.FeatureAlphaDropout"),
  std::make_tuple(cs("flatten"),          Cast::flatten,         typeid(torch::nn::FlattenImpl).hash_code(),             "torch.nn.Flatten"),
  std::make_tuple(cs("fmaxpool2d"),       Cast::fmaxpool2d,      typeid(torch::nn::FractionalMaxPool2dImpl).hash_code(), "torch.nn.FractionalMaxPool2d"),
  std::make_tuple(cs("fmaxpool3d"),       Cast::fmaxpool3d,      typeid(torch::nn::FractionalMaxPool3dImpl).hash_code(), "torch.nn.FractionalMaxPool3d"),
  std::make_tuple(cs("fold"),             Cast::fold,            typeid(torch::nn::FoldImpl).hash_code(),                "torch.nn.Fold"),
  std::make_tuple(cs("fork"),             Cast::fork,            typeid(ForkImpl).hash_code(),                           ""),
  std::make_tuple(cs("gelu"),             Cast::gelu,            typeid(torch::nn::GELUImpl).hash_code(),                "torch.nn.GELU"),
  std::make_tuple(cs("glu"),              Cast::glu,             typeid(torch::nn::GLUImpl).hash_code(),                 "torch.nn.functional.glu"),
  std::make_tuple(cs("groupnorm"),        Cast::groupnorm,       typeid(torch::nn::GroupNormImpl).hash_code(),           "torch.nn.GroupNorm"),
  std::make_tuple(cs("gru"),              Cast::gru,             typeid(torch::nn::GRUImpl).hash_code(),                 "torch.nn.GRU"),
  std::make_tuple(cs("gruout"),           Cast::gruout,          typeid(GRUOutputImpl).hash_code(),                      "torch.nn.GRU"),
  std::make_tuple(cs("hardshrink"),       Cast::hardshrink,      typeid(torch::nn::HardshrinkImpl).hash_code(),          "torch.nn.Hardshrink"),
  std::make_tuple(cs("hardtanh"),         Cast::hardtanh,        typeid(torch::nn::HardtanhImpl).hash_code(),            "torch.nn.Hardtanh"),
  std::make_tuple(cs("identity"),         Cast::identity,        typeid(torch::nn::IdentityImpl).hash_code(),            "torch.nn.Identity"),
  std::make_tuple(cs("indexselect"),      Cast::indexselect,     typeid(IndexSelectImpl).hash_code(),                    "torch.Tensor.index_select"),
  std::make_tuple(cs("instancenorm1d"),   Cast::instancenorm1d,  typeid(torch::nn::InstanceNorm1dImpl).hash_code(),      "torch.nn.InstanceNorm1d"),
  std::make_tuple(cs("instancenorm2d"),   Cast::instancenorm2d,  typeid(torch::nn::InstanceNorm2dImpl).hash_code(),      "torch.nn.InstanceNorm2d"),
  std::make_tuple(cs("instancenorm3d"),   Cast::instancenorm3d,  typeid(torch::nn::InstanceNorm3dImpl).hash_code(),      "torch.nn.InstanceNorm3d"),
  std::make_tuple(cs("interpolate"),      Cast::interpolate,     typeid(torch::nn::functional::interpolate).hash_code(), "torch.nn.functional.interpolate"),
  std::make_tuple(cs("layernorm"),        Cast::layernorm,       typeid(torch::nn::LayerNormImpl).hash_code(),           "torch.nn.LayerNorm"),
  std::make_tuple(cs("leakyrelu"),        Cast::leakyrelu,       typeid(torch::nn::LeakyReLUImpl).hash_code(),           "torch.nn.LeakyReLU"),
  std::make_tuple(cs("linear"),           Cast::linear,          typeid(torch::nn::LinearImpl).hash_code(),              "torch.nn.Linear"),
  std::make_tuple(cs("localnorm"),        Cast::localnorm,       typeid(torch::nn::LocalResponseNormImpl).hash_code(),   "torch.nn.LocalResponseNorm"),
  std::make_tuple(cs("logsigmoid"),       Cast::logsigmoid,      typeid(torch::nn::LogSigmoidImpl).hash_code(),          "torch.nn.LogSigmoid"),
  std::make_tuple(cs("logsoftmax"),       Cast::logsoftmax,      typeid(torch::nn::LogSoftmaxImpl).hash_code(),          "torch.nn.LogSoftmax"),
  std::make_tuple(cs("lppool1d"),         Cast::lppool1d,        typeid(torch::nn::LPPool1dImpl).hash_code(),            "torch.nn.LPPool1d"),
  std::make_tuple(cs("lppool2d"),         Cast::lppool2d,        typeid(torch::nn::LPPool2dImpl).hash_code(),            "torch.nn.LPPool2d"),
  std::make_tuple(cs("lstm"),             Cast::lstm,            typeid(LSTMImpl).hash_code(),                           "torch.nn.LSTM"),
  std::make_tuple(cs("lstmout"),          Cast::lstmout,         typeid(LSTMOutputImpl).hash_code(),                     "torch.nn.LSTM"),
  std::make_tuple(cs("maxpool1d"),        Cast::maxpool1d,       typeid(torch::nn::MaxPool1dImpl).hash_code(),           "torch.nn.MaxPool1d"),
  std::make_tuple(cs("maxpool2d"),        Cast::maxpool2d,       typeid(torch::nn::MaxPool2dImpl).hash_code(),           "torch.nn.MaxPool2d"),
  std::make_tuple(cs("maxpool3d"),        Cast::maxpool3d,       typeid(torch::nn::MaxPool3dImpl).hash_code(),           "torch.nn.MaxPool3d"),
  std::make_tuple(cs("mish"),             Cast::mish,            typeid(torch::nn::MishImpl).hash_code(),                "torch.nn.Mish"),
  std::make_tuple(cs("moduledict"),       Cast::moduledict,      typeid(torch::nn::ModuleDictImpl).hash_code(),          "torch.nn.ModuleDict"),
  std::make_tuple(cs("modulelist"),       Cast::modulelist,      typeid(torch::nn::ModuleListImpl).hash_code(),          "torch.nn.ModuleList"),
  std::make_tuple(cs("mul"),              Cast::mul,             typeid(MulImpl).hash_code(),                            "torch.mul"),
  std::make_tuple(cs("nbeats"),           Cast::nbeats,          typeid(NBeatsImpl).hash_code(),                         ""),
  std::make_tuple(cs("normalize"),        Cast::normalize,       typeid(torch::nn::functional::normalize).hash_code(),   "torch.nn.functional.normalize"),
  std::make_tuple(cs("onehot"),           Cast::onehot,          typeid(OneHotImpl).hash_code(),                         "torch.nn.functional.one_hot"),
  std::make_tuple(cs("pad"),              Cast::pad,             typeid(PadImpl).hash_code(),                            "torch.nn.functional.pad"),
  std::make_tuple(cs("pad1d"),            Cast::pad1d,           typeid(torch::nn::ConstantPad1dImpl).hash_code(),       "torch.nn.ConstantPad1d"),
  std::make_tuple(cs("pad2d"),            Cast::pad2d,           typeid(torch::nn::ConstantPad2dImpl).hash_code(),       "torch.nn.ConstantPad2d"),
  std::make_tuple(cs("pad3d"),            Cast::pad3d,           typeid(torch::nn::ConstantPad3dImpl).hash_code(),       "torch.nn.ConstantPad3d"),
  std::make_tuple(cs("pairwise"),         Cast::pairwise,        typeid(torch::nn::PairwiseDistanceImpl).hash_code(),    "torch.nn.PairwiseDistance"),
  std::make_tuple(cs("parmdict"),         Cast::parmdict,        typeid(torch::nn::ParameterDictImpl).hash_code(),       "torch.nn.ParameterDict"),
  std::make_tuple(cs("prelu"),            Cast::prelu,           typeid(torch::nn::PReLUImpl).hash_code(),               "torch.nn.PReLU"),
  std::make_tuple(cs("randomcrop"),       Cast::randomcrop,      typeid(RandomCropImpl).hash_code(),                     ""),
  std::make_tuple(cs("randomflip"),       Cast::randomflip,      typeid(RandomFlipImpl).hash_code(),                     ""),
  std::make_tuple(cs("recur"),            Cast::recur,           typeid(RecurImpl).hash_code(),                          ""),
  std::make_tuple(cs("reflect1d"),        Cast::reflect1d,       typeid(torch::nn::ReflectionPad1dImpl).hash_code(),     "torch.nn.ReflectionPad1d"),
  std::make_tuple(cs("reflect2d"),        Cast::reflect2d,       typeid(torch::nn::ReflectionPad2dImpl).hash_code(),     "torch.nn.ReflectionPad2d"),
  std::make_tuple(cs("relu"),             Cast::relu,            typeid(torch::nn::ReLUImpl).hash_code(),                "torch.nn.ReLU"),
  std::make_tuple(cs("relu6"),            Cast::relu6,           typeid(torch::nn::ReLU6Impl).hash_code(),               "torch.nn.ReLU6"),
  std::make_tuple(cs("replicate1d"),      Cast::replicate1d,     typeid(torch::nn::ReplicationPad1dImpl).hash_code(),    "torch.nn.ReplicationPad1d"),
  std::make_tuple(cs("replicate2d"),      Cast::replicate2d,     typeid(torch::nn::ReplicationPad2dImpl).hash_code(),    "torch.nn.ReplicationPad2d"),
  std::make_tuple(cs("replicate3d"),      Cast::replicate3d,     typeid(torch::nn::ReplicationPad3dImpl).hash_code(),    "torch.nn.ReplicationPad3d"),
  std::make_tuple(cs("residual"),         Cast::residual,        typeid(ResidualImpl).hash_code(),                       ""),
  std::make_tuple(cs("reshape"),          Cast::reshape,         typeid(ReshapeImpl).hash_code(),                        "torch.reshape"),
  std::make_tuple(cs("rnn"),              Cast::rnn,             typeid(torch::nn::RNNImpl).hash_code(),                 "torch.nn.RNN"),
  std::make_tuple(cs("rnnout"),           Cast::rnnout,          typeid(RNNOutputImpl).hash_code(),                      "torch.nn.RNN"),
  std::make_tuple(cs("rrelu"),            Cast::rrelu,           typeid(torch::nn::RReLUImpl).hash_code(),               "torch.nn.RReLU"),
  std::make_tuple(cs("select"),           Cast::select,          typeid(SelectImpl).hash_code(),                         "torch.Tensor.select"),
  std::make_tuple(cs("selu"),             Cast::selu,            typeid(torch::nn::SELUImpl).hash_code(),                "torch.nn.SELU"),
  std::make_tuple(cs("seqjoin"),          Cast::seqjoin,         typeid(SeqJoinImpl).hash_code(),                        ""),
  std::make_tuple(cs("seqnest"),          Cast::seqnest,         typeid(SeqNestImpl).hash_code(),                        ""),
  std::make_tuple(cs("sequential"),       Cast::sequential,      typeid(torch::nn::SequentialImpl).hash_code(),          "torch.nn.Sequential"),
  std::make_tuple(cs("sigmoid"),          Cast::sigmoid,         typeid(torch::nn::SigmoidImpl).hash_code(),             "torch.nn.Sigmoid"),
  std::make_tuple(cs("silu"),             Cast::silu,            typeid(torch::nn::SiLUImpl).hash_code(),                "torch.nn.SiLU"),
  std::make_tuple(cs("similar"),          Cast::similar,         typeid(torch::nn::CosineSimilarityImpl).hash_code(),    "torch.nn.CosineSimilarity"),
  std::make_tuple(cs("softmax"),          Cast::softmax,         typeid(torch::nn::SoftmaxImpl).hash_code(),             "torch.nn.Softmax"),
  std::make_tuple(cs("softmax2d"),        Cast::softmax2d,       typeid(torch::nn::Softmax2dImpl).hash_code(),           "torch.nn.Softmax2d"),
  std::make_tuple(cs("softmin"),          Cast::softmin,         typeid(torch::nn::SoftminImpl).hash_code(),             "torch.nn.Softmin"),
  std::make_tuple(cs("softplus"),         Cast::softplus,        typeid(torch::nn::SoftplusImpl).hash_code(),            "torch.nn.Softplus"),
  std::make_tuple(cs("softshrink"),       Cast::softshrink,      typeid(torch::nn::SoftshrinkImpl).hash_code(),          "torch.nn.Softshrink"),
  std::make_tuple(cs("softsign"),         Cast::softsign,        typeid(torch::nn::SoftsignImpl).hash_code(),            "torch.nn.Softsign"),
  std::make_tuple(cs("squeeze"),          Cast::squeeze,         typeid(SqueezeImpl).hash_code(),                        "torch.squeeze"),
  std::make_tuple(cs("tanh"),             Cast::tanh,            typeid(torch::nn::TanhImpl).hash_code(),                "torch.nn.Tanh"),
  std::make_tuple(cs("tanhshrink"),       Cast::tanhshrink,      typeid(torch::nn::TanhshrinkImpl).hash_code(),          "torch.nn.Tanhshrink"),
  std::make_tuple(cs("threshold"),        Cast::threshold,       typeid(torch::nn::ThresholdImpl).hash_code(),           "torch.nn.Threshold"),
  std::make_tuple(cs("transform"),        Cast::transform,       typeid(TransformImpl).hash_code(),                      ""),
  std::make_tuple(cs("transformer"),      Cast::transformer,     typeid(torch::nn::TransformerImpl).hash_code(),         "torch.nn.Transformer"),
  std::make_tuple(cs("unfold"),           Cast::unfold,          typeid(torch::nn::UnfoldImpl).hash_code(),              "torch.nn.Unfold"),
  std::make_tuple(cs("unsqueeze"),        Cast::unsqueeze,       typeid(UnsqueezeImpl).hash_code(),                      "torch.unsqueeze"),
  std::make_tuple(cs("upsample"),         Cast::upsample,        typeid(torch::nn::UpsampleImpl).hash_code(),            "torch.nn.Upsample"),
  std::make_tuple(cs("zeropad2d"),        Cast::zeropad2d,       typeid(torch::nn::ZeroPad2dImpl).hash_code(),           "torch.nn.ZeroPad2d"),
  std::make_tuple(cs("zscore"),           Cast::zscore,          typeid(ZscoreImpl).hash_code(),                         "")
 }};

 std::array<std::tuple<S,Setting>,15> cset = {{            // configuration settings
 std::make_tuple(cs("mkl"),                Setting::mkl),
 std::make_tuple(cs("openmp"),             Setting::openmp),
 std::make_tuple(cs("threads"),            Setting::threads),
 std::make_tuple(cs("interopthreads"),     Setting::interopthreads),
 std::make_tuple(cs("cuda"),               Setting::cuda),
 std::make_tuple(cs("magma"),              Setting::magma),
 std::make_tuple(cs("cudnn"),              Setting::cudnn),
 std::make_tuple(cs("cudnnversion"),       Setting::cudnnversion),
 std::make_tuple(cs("cudadevices"),        Setting::cudadevices),
 std::make_tuple(cs("benchmark"),          Setting::benchmark),
 std::make_tuple(cs("deterministic"),      Setting::deterministic),
 std::make_tuple(cs("cudnndeterministic"), Setting::cudnndeterministic),
 std::make_tuple(cs("stackframe"),         Setting::stackframe),
 std::make_tuple(cs("alloptions"),         Setting::alloptions),
 std::make_tuple(cs("complexfirst"),       Setting::complexfirst)
}};

 std::array<std::tuple<S,Setting>,84> mset = {{        // module option sym -> enum
  std::make_tuple(cs("addbias"),      Setting::addbias),
  std::make_tuple(cs("addzero"),      Setting::addzero),
  std::make_tuple(cs("affine"),       Setting::affine),
  std::make_tuple(cs("alpha"),        Setting::alpha),
  std::make_tuple(cs("align"),        Setting::align),
  std::make_tuple(cs("batchfirst"),   Setting::batchfirst),
  std::make_tuple(cs("beta"),         Setting::beta),
  std::make_tuple(cs("bi"),           Setting::bi),
  std::make_tuple(cs("bias"),         Setting::bias),
  std::make_tuple(cs("ceiling"),      Setting::ceiling),
  std::make_tuple(cs("channels"),     Setting::channels),
  std::make_tuple(cs("classes"),      Setting::classes),
  std::make_tuple(cs("cols"),         Setting::cols),
  std::make_tuple(cs("countpad"),     Setting::countpad),
  std::make_tuple(cs("decoder"),      Setting::decoder),
  std::make_tuple(cs("decoderlayer"), Setting::decoderlayer),
  std::make_tuple(cs("detach"),       Setting::detach),
  std::make_tuple(cs("dilate"),       Setting::dilate),
  std::make_tuple(cs("divisor"),      Setting::divisor),
  std::make_tuple(cs("dim"),          Setting::dim),
  std::make_tuple(cs("dlayers"),      Setting::dlayers),
  std::make_tuple(cs("dropout"),      Setting::dropout),
  std::make_tuple(cs("dtype"),        Setting::dtype),
  std::make_tuple(cs("elayers"),      Setting::elayers),
  std::make_tuple(cs("encoder"),      Setting::encoder),
  std::make_tuple(cs("encoderlayer"), Setting::encoderlayer),
  std::make_tuple(cs("end"),          Setting::end),
  std::make_tuple(cs("eps"),          Setting::eps),
  std::make_tuple(cs("fn"),           Setting::fn),
  std::make_tuple(cs("freeze"),       Setting::freeze),
  std::make_tuple(cs("groups"),       Setting::groups),
  std::make_tuple(cs("heads"),        Setting::heads),
  std::make_tuple(cs("hidden"),       Setting::hidden),
  std::make_tuple(cs("in"),           Setting::in),
  std::make_tuple(cs("in1"),          Setting::in1),
  std::make_tuple(cs("in2"),          Setting::in2),
  std::make_tuple(cs("ind"),          Setting::ind),
  std::make_tuple(cs("indices"),      Setting::indices),
  std::make_tuple(cs("init"),         Setting::init),
  std::make_tuple(cs("inplace"),      Setting::inplace),
  std::make_tuple(cs("k"),            Setting::k),
  std::make_tuple(cs("kdim"),         Setting::kdim),
  std::make_tuple(cs("keepdim"),      Setting::keepdim),
  std::make_tuple(cs("kvbias"),       Setting::kvbias),
  std::make_tuple(cs("kvzeros"),      Setting::kvzeros),
  std::make_tuple(cs("lambda"),       Setting::lambda),
  std::make_tuple(cs("lastoffset"),   Setting::lastoffset),
  std::make_tuple(cs("layers"),       Setting::layers),
  std::make_tuple(cs("layernorm"),    Setting::layernorm),
  std::make_tuple(cs("lower"),        Setting::lower),
  std::make_tuple(cs("max"),          Setting::max),
  std::make_tuple(cs("maxnorm"),      Setting::maxnorm),
  std::make_tuple(cs("mean"),         Setting::mean),
  std::make_tuple(cs("min"),          Setting::min),
  std::make_tuple(cs("mode"),         Setting::mode),
  std::make_tuple(cs("momentum"),     Setting::momentum),
  std::make_tuple(cs("norm"),         Setting::norm),
  std::make_tuple(cs("out"),          Setting::out),
  std::make_tuple(cs("outpad"),       Setting::outpad),
  std::make_tuple(cs("outsize"),      Setting::outsize),
  std::make_tuple(cs("p"),            Setting::p),
  std::make_tuple(cs("pad"),          Setting::pad),
  std::make_tuple(cs("padflag"),      Setting::padflag),
  std::make_tuple(cs("padindex"),     Setting::padindex),
  std::make_tuple(cs("padmode"),      Setting::padmode),
  std::make_tuple(cs("ratio"),        Setting::ratio),
  std::make_tuple(cs("rescale"),      Setting::rescale),
  std::make_tuple(cs("rows"),         Setting::rows),
  std::make_tuple(cs("scale"),        Setting::scale),
  std::make_tuple(cs("size"),         Setting::size),
  std::make_tuple(cs("shape"),        Setting::shape),
  std::make_tuple(cs("slope"),        Setting::slope),
  std::make_tuple(cs("sparse"),       Setting::sparse),
  std::make_tuple(cs("start"),        Setting::start),
  std::make_tuple(cs("std"),          Setting::std),
  std::make_tuple(cs("stride"),       Setting::stride),
  std::make_tuple(cs("threshold"),    Setting::threshold),
  std::make_tuple(cs("track"),        Setting::track),
  std::make_tuple(cs("train"),        Setting::train),
  std::make_tuple(cs("transpose"),    Setting::transpose),
  std::make_tuple(cs("upper"),        Setting::upper),
  std::make_tuple(cs("value"),        Setting::value),
  std::make_tuple(cs("vdim"),         Setting::vdim),
  std::make_tuple(cs("weight"),       Setting::weight)
 }};

 std::array<std::tuple<S,State>,11> state = {{  //module state dictionary keys: map symbol -> enum
  std::make_tuple(cs("buffers"),   State::buffers),
  std::make_tuple(cs("depth"),     State::depth),
  std::make_tuple(cs("module"),    State::module),
  std::make_tuple(cs("name"),      State::name),
  std::make_tuple(cs("options"),   State::options),
  std::make_tuple(cs("options"),   State::optlist),
  std::make_tuple(cs("optimizer"), State::optimizer),
  std::make_tuple(cs("parms"),     State::parms),
  std::make_tuple(cs("parmgroup"), State::parmgroup),
  std::make_tuple(cs("pointer"),   State::pointer),
  std::make_tuple(cs("size"),      State::size)
 }};

 std::array<std::tuple<S,Cast,std::string>,22> loss = {{   // loss: map symbol -> enum
  std::make_tuple(cs("bce"),         Cast::bce,         "torch.nn.BCELoss"),
  std::make_tuple(cs("bcelogits"),   Cast::bcelogits,   "torch.nn.BCEWithLogitsLoss"),
  std::make_tuple(cs("ce"),          Cast::ce,          "torch.nn.CrossEntropyLoss"),
  std::make_tuple(cs("cosineloss"),  Cast::cosineloss,  "torch.nn.CosineEmbeddingLoss"),
  std::make_tuple(cs("ctc"),         Cast::ctc,         "torch.nn.CTCLoss"),
  std::make_tuple(cs("hinge"),       Cast::hinge,       "torch.nn.HingeEmbeddingLoss"),
  std::make_tuple(cs("huber"),       Cast::huber,       "torch.nn.HuberLoss"),
  std::make_tuple(cs("kl"),          Cast::kl,          "torch.nn.KLDivLoss"),
  std::make_tuple(cs("l1"),          Cast::l1,          "torch.nn.L1Loss"),
  std::make_tuple(cs("margin"),      Cast::margin,      "torch.nn.MarginRankingLoss"),
  std::make_tuple(cs("mse"),         Cast::mse,         "torch.nn.MSELoss"),
  std::make_tuple(cs("multilabel"),  Cast::multilabel,  "torch.nn.MultiLabelMarginLoss"),
  std::make_tuple(cs("multimargin"), Cast::multimargin, "torch.nn.MultiMarginLoss"),
  std::make_tuple(cs("multisoft"),   Cast::multisoft,   "torch.nn.MultiLabelSoftMarginLoss"),
  std::make_tuple(cs("nll"),         Cast::nll,         "torch.nn.NLLLoss"),
  std::make_tuple(cs("pairwise"),    Cast::pairwise,    "torch.nn.PairwiseDistance"),
  std::make_tuple(cs("poissonloss"), Cast::poissonloss, "torch.nn.PoissonNLLLoss"),
  std::make_tuple(cs("sce"),         Cast::sce,         "SmoothCrossEntropy"),
  std::make_tuple(cs("similar"),     Cast::similar,     "torch.nn.CosineSimilarity"),
  std::make_tuple(cs("smoothl1"),    Cast::smoothl1,    "torch.nn.SmoothL1Loss"),
  std::make_tuple(cs("softmargin"),  Cast::softmargin,  "torch.nn.SoftMarginLoss"),
  std::make_tuple(cs("triplet"),     Cast::triplet,     "torch.nn.TripletMarginLoss")
 }};

 std::array<std::tuple<S,Setting>,13> lset = {{     // loss option sym -> enum
  std::make_tuple(cs("blank"),     Setting::blank),
  std::make_tuple(cs("eps"),       Setting::eps),
  std::make_tuple(cs("delta"),     Setting::delta),
  std::make_tuple(cs("full"),      Setting::full),
  std::make_tuple(cs("ignore"),    Setting::ignore),
  std::make_tuple(cs("log"),       Setting::log),
  std::make_tuple(cs("margin"),    Setting::margin),
  std::make_tuple(cs("p"),         Setting::p),
  std::make_tuple(cs("reduce"),    Setting::reduce),
  std::make_tuple(cs("smoothing"), Setting::smoothing),
  std::make_tuple(cs("swap"),      Setting::swap),
  std::make_tuple(cs("weight"),    Setting::weight),
  std::make_tuple(cs("zeroinf"),   Setting::zeroinf)
 }};

 std::array<std::tuple<S,Cast,std::string>,6> opt = {{        //optimizer: map symbol -> enum
  std::make_tuple(cs("adagrad"), Cast::adagrad, "torch.optim.Adagrad"),
  std::make_tuple(cs("adam"),    Cast::adam,    "torch.optim.Adam"),
  std::make_tuple(cs("adamw"),   Cast::adamw,   "torch.optim.AdamW"),
  std::make_tuple(cs("lbfgs"),   Cast::lbfgs,   "torch.optim.LBFGS"),
  std::make_tuple(cs("rmsprop"), Cast::rmsprop, "torch.optim.RMSprop"),
  std::make_tuple(cs("sgd"),     Cast::sgd,     "torch.optim.SGD")
 }};

 std::array<std::tuple<S,Setting>,19> oset = {{         //optimizer setting: map symbol -> enum
  std::make_tuple(cs("alpha"),      Setting::alpha),
  std::make_tuple(cs("amsgrad"),    Setting::amsgrad),
  std::make_tuple(cs("beta1"),      Setting::beta1),
  std::make_tuple(cs("beta2"),      Setting::beta2),
  std::make_tuple(cs("centered"),   Setting::centered),
  std::make_tuple(cs("changetol"),  Setting::changetol),
  std::make_tuple(cs("dampening"),  Setting::dampening),
  std::make_tuple(cs("decay"),      Setting::decay),
  std::make_tuple(cs("eps"),        Setting::eps),
  std::make_tuple(cs("eval"),       Setting::eval),
  std::make_tuple(cs("gradtol"),    Setting::gradtol),
  std::make_tuple(cs("history"),    Setting::history),
  std::make_tuple(cs("init"),       Setting::init),
  std::make_tuple(cs("iter"),       Setting::iter),
  std::make_tuple(cs("lrdecay"),    Setting::lrdecay),
  std::make_tuple(cs("lr"),         Setting::lr),
  std::make_tuple(cs("momentum"),   Setting::momentum),
  std::make_tuple(cs("nesterov"),   Setting::nesterov),
  std::make_tuple(cs("search"),     Setting::search)
 }};

 std::array<std::tuple<S,Attr>,35> attr = {{            //attributes: map symbol -> enum
  std::make_tuple(cs("bytes"),        Attr::bytes),
  std::make_tuple(cs("coalesced"),    Attr::coalesced),
  std::make_tuple(cs("contiguous"),   Attr::contiguous),
  std::make_tuple(cs("contiguous2d"), Attr::contiguous2d),
  std::make_tuple(cs("contiguous3d"), Attr::contiguous3d),
  std::make_tuple(cs("data"),         Attr::data),
  std::make_tuple(cs("defined"),      Attr::defined),
  std::make_tuple(cs("device"),       Attr::device),
  std::make_tuple(cs("densedim"),     Attr::densedim),
  std::make_tuple(cs("dim"),          Attr::dim),
  std::make_tuple(cs("dtype"),        Attr::dtype),
  std::make_tuple(cs("elements"),     Attr::elements),
  std::make_tuple(cs("gradflag"),     Attr::gradflag),
  std::make_tuple(cs("gradfn"),       Attr::gradfn),
  std::make_tuple(cs("gradient"),     Attr::gradient),
  std::make_tuple(cs("itemsize"),     Attr::itemsize),
  std::make_tuple(cs("layout"),       Attr::layout),
  std::make_tuple(cs("leaf"),         Attr::leaf),
  std::make_tuple(cs("memory"),       Attr::memory),
  std::make_tuple(cs("nnz"),          Attr::nnz),
  std::make_tuple(cs("numel"),        Attr::numel),
  std::make_tuple(cs("offset"),       Attr::offset),
  std::make_tuple(cs("pin"),          Attr::pinned),
  std::make_tuple(cs("ptr"),          Attr::ptr),
  std::make_tuple(cs("ref"),          Attr::ref),
  std::make_tuple(cs("result"),       Attr::result),
  std::make_tuple(cs("size"),         Attr::size),
  std::make_tuple(cs("sparseflag"),   Attr::sparseflag),
  std::make_tuple(cs("sparsedim"),    Attr::sparsedim),
  std::make_tuple(cs("sptr"),         Attr::sptr),
  std::make_tuple(cs("sref"),         Attr::sref),
  std::make_tuple(cs("storage"),      Attr::storage),
  std::make_tuple(cs("stride"),       Attr::stride),
  std::make_tuple(cs("tensorcount"),  Attr::tensorcount),
  std::make_tuple(cs("weakref"),      Attr::weakref)
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
 std::array<std::tuple<S,Enum>,35> enums = {{
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
  std::make_tuple(cs("gelu"),            Enum::gelu),
  std::make_tuple(cs("leakyrelu"),       Enum::leakyrelu),
  std::make_tuple(cs("linear"),          Enum::linear),
  std::make_tuple(cs("max"),             Enum::max),
  std::make_tuple(cs("mean"),            Enum::mean),
  std::make_tuple(cs("mish"),            Enum::mish),
  std::make_tuple(cs("nearest"),         Enum::nearest),
  std::make_tuple(cs("none"),            Enum::none),
  std::make_tuple(cs("reflect"),         Enum::reflect),
  std::make_tuple(cs("reflection"),      Enum::reflection),
  std::make_tuple(cs("relu"),            Enum::relu),
  std::make_tuple(cs("replicate"),       Enum::replicate),
  std::make_tuple(cs("same"),            Enum::same),
  std::make_tuple(cs("sigmoid"),         Enum::sigmoid),
  std::make_tuple(cs("silu"),            Enum::silu),
  std::make_tuple(cs("sum"),             Enum::sum),
  std::make_tuple(cs("tanh"),            Enum::tanh),
  std::make_tuple(cs("trilinear"),       Enum::trilinear),
  std::make_tuple(cs("valid"),           Enum::valid),
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
 S operator()(const torch::enumtype::kGELU&)             const { return std::get<0>(env().enums[(size_t)Enum::gelu]);}
 S operator()(const torch::enumtype::kLeakyReLU&)        const { return std::get<0>(env().enums[(size_t)Enum::leakyrelu]);}
 S operator()(const torch::enumtype::kLinear&)           const { return std::get<0>(env().enums[(size_t)Enum::linear]);}
 S operator()(const torch::enumtype::kMax&)              const { return std::get<0>(env().enums[(size_t)Enum::max]);}
 S operator()(const torch::enumtype::kMean&)             const { return std::get<0>(env().enums[(size_t)Enum::mean]);}
 S operator()(const torch::enumtype::kMish&)             const { return std::get<0>(env().enums[(size_t)Enum::mish]);}
 S operator()(const torch::enumtype::kNearest&)          const { return std::get<0>(env().enums[(size_t)Enum::nearest]);}
 S operator()(const torch::enumtype::kNone&)             const { return std::get<0>(env().enums[(size_t)Enum::none]);}
 S operator()(const torch::enumtype::kReflect&)          const { return std::get<0>(env().enums[(size_t)Enum::reflect]);}
 S operator()(const torch::enumtype::kReflection&)       const { return std::get<0>(env().enums[(size_t)Enum::reflection]);}
 S operator()(const torch::enumtype::kReLU&)             const { return std::get<0>(env().enums[(size_t)Enum::relu]);}
 S operator()(const torch::enumtype::kReplicate&)        const { return std::get<0>(env().enums[(size_t)Enum::replicate]);}
 S operator()(const torch::enumtype::kSame&)             const { return std::get<0>(env().enums[(size_t)Enum::same]);}
 S operator()(const torch::enumtype::kSigmoid&)          const { return std::get<0>(env().enums[(size_t)Enum::sigmoid]);}
 S operator()(const torch::enumtype::kSiLU&)             const { return std::get<0>(env().enums[(size_t)Enum::silu]);}
 S operator()(const torch::enumtype::kSum&)              const { return std::get<0>(env().enums[(size_t)Enum::sum]);}
 S operator()(const torch::enumtype::kTanh&)             const { return std::get<0>(env().enums[(size_t)Enum::tanh]);}
 S operator()(const torch::enumtype::kTrilinear&)        const { return std::get<0>(env().enums[(size_t)Enum::trilinear]);}
 S operator()(const torch::enumtype::kValid&)            const { return std::get<0>(env().enums[(size_t)Enum::valid]);}
 S operator()(const torch::enumtype::kZeros&)            const { return std::get<0>(env().enums[(size_t)Enum::zeros]);}
};

Esym& esym();
#define ESYM(v) c10::visit(esym(), v)
