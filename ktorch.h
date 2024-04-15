#pragma once

#ifdef __clang__
# pragma clang diagnostic push
# pragma GCC diagnostic ignored "-Wgnu-anonymous-struct"                   // k.h warning
# pragma GCC diagnostic ignored "-Wnested-anon-types"                      // k.h warning
# pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"    // ATen.h VA_ARG warning, FORWARD_HAS_DEFAULT_ARGS
# pragma clang diagnostic ignored "-Wunused-function"                      // private.h generates 'unused function' warnings
# pragma clang diagnostic ignored "-Wc++1z-extensions"                     // nodiscard & fallthrough warnings
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
#define KSHORT 5
#undef KH  // conflict introduced with v1.10.0, include/ATen/ops/avg_pool2d_meta.h: template <bool KH..

#include <stack>
#include "torch/torch.h"
#include "private.h"

// access private name_ & buffers_ of Module
using Tensor     = torch::Tensor;
using TensorDict = torch::OrderedDict<std::string, Tensor>;
using Module     = torch::nn::Module;
using Moduleptr  = std::shared_ptr<Module>;
using Modulemap  = torch::OrderedDict<std::string, Moduleptr>;
using Generator  = torch::Generator;
ACCESS_PRIVATE_FIELD(Module, c10::optional<std::string>, name_)
ACCESS_PRIVATE_FIELD(Module, TensorDict, buffers_)
ACCESS_PRIVATE_FIELD(Module, TensorDict, parameters_)
ACCESS_PRIVATE_FIELD(Module, Modulemap,  children_)

#ifdef __clang__
# pragma clang diagnostic pop
#elif defined __GNUC__
# pragma GCC diagnostic pop
#endif

#define TORCH_ERROR(...)                                                    \
 do {                                                                       \
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
#define KEX(x) kexpand(x.size(),(*x).data())  // k list from ExpandingArray
#define ESYM(v) std::visit(esym(), v)

using Ktype=signed char;
using Device=torch::Device;
using DeviceType=torch::DeviceType;
using Storage=torch::Storage;
using Scalar=torch::Scalar;
using TensorVector=std::vector<Tensor>;
using TensorDeque=std::deque<Tensor>;
using LongVector=std::vector<int64_t>;
using DoubleVector=std::vector<double>;
using IntArrayRef=torch::IntArrayRef;
using SymArrayRef=torch::ArrayRef<S>;
using DoubleArrayRef=torch::ArrayRef<double>;
template<size_t D,typename T=int64_t> using ExpandingArray=torch::ExpandingArray<D,T>;
template<size_t D,typename T=double>  using Exdouble=torch::ExpandingArray<D,T>;
template<size_t D,typename T=int64_t> using Exoptional=torch::ExpandingArrayWithOptionalElem<D,T>;
using Dtype=torch::Dtype;
using TypeMeta=caffe2::TypeMeta;
using ScalarType=torch::ScalarType;
using TensorOptions=torch::TensorOptions;
using TensorList=torch::TensorList;  // ArrayRef<Tensor>

// shorter names for commonly used module structures
using Modules     = std::stack<Moduleptr>;
using Modulepairs = std::vector<std::pair<std::string, std::shared_ptr<Module>>>;
using AnyModule   = torch::nn::AnyModule;

struct Empty{};
using Tuple       = std::tuple<Tensor,Tensor>;
using Tuple3      = std::tuple<Tensor,Tensor,Tensor>;
using Tuple4      = std::tuple<Tensor,Tensor,Tensor,Tensor>;
using Nested      = std::tuple<Tensor,Tuple>;
using Input       = std::variant<Tensor,TensorVector,TensorDict,Empty>; 
using Output      = std::variant<Tensor,Tuple,Nested,TensorVector>;
using MetricData  = std::vector<TensorVector>;

using Optimizer   = torch::optim::Optimizer;
using Optptr      = std::shared_ptr<Optimizer>;

typedef struct Pairs {
 Ktype a = 0;  // type: 1-dict, 2-list of pairs, 3-general list, 4-sym list
 Ktype t = 0;  // type of value in last pair processed
 H i = 0;  // next pair to process
 H n = 0;  // count of pairs
 S k = 0;  // name of current name,value pair
 K x = 0;  // k value with dict/pairs/list
 union {
  bool   b;  // boolean value from current pair
  J      j;  // long value
  float  e;  // float value
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
 loss,
 optimizer,
 model,
 train,
 test
};

enum class Arg:short {  // type of input(s) & output for callback modules
 undefined=0,
 boolean,
 tensor,
 tuple,
 nested,
 vector,
 dict
};

enum class Cast:short {
 undefined=0, 
 tensor,          parameter,       buffer,          model,      // basic structures
 callback,        moduledict,      modulelist,      parmdict,   // container modules
 sequential,      seqdict,         seqlist,         seqnest,        seqjoin,

 adaptavg1d,      adaptavg2d,      adaptavg3d,      adaptmax1d,     adaptmax2d,  // modules
 adaptmax3d,      adrop,           attention,       avgpool1d,      avgpool2d,
 avgpool3d,       batchnorm1d,     batchnorm2d,     batchnorm3d,    bilinear,
 cat,             celu,            conv1d,          conv2d,         conv3d,
 convtranspose1d, convtranspose2d, convtranspose3d, crossmap2d,     decoder,
 decoderlayer,    drop,            drop2d,          drop3d,         droppath,
 elu,             embed,           embedbag,        embedpos,       embedseq,
 encoder,         encoderlayer,    expand,          fadrop,         flatten,
 fmaxpool2d,      fmaxpool3d,      fold,            fork,           gelu,
 glu,             groupnorm,       gru,             hardshrink,     hardtanh,
 identity,        indexselect,     instancenorm1d,  instancenorm2d, instancenorm3d,
 interpolate,     layernorm,       leakyrelu,       linear,         localnorm,
 logsigmoid,      logsoftmax,      lppool1d,        lppool2d,       lstm,
 matmul,          maxpool1d,       maxpool2d,       maxpool3d,      mish,
 mul,             nbeats,          normalize,       onehot,         pad,
 pad1d,           pad2d,           pad3d,           permute,        prelu,
 randomcrop,      randomflip,      recur,           reflect1d,      reflect2d,
 relu,            relu6,           replicate1d,     replicate2d,    replicate3d,
 reshape,         residual,        rnn,             rrelu,          select,
 selfattention,   selu,            sigmoid,         silu,           softmax,
 softmax2d,       softmin,         softplus,        softshrink,     softsign,
 squeeze,         tanh,            tanhshrink,      threshold,      transform,
 transformer,     transpose,       unfold,          unsqueeze,      upsample,
 zeropad2d,       zscore,

 pairwise,    similar,                                                                 // distance functions
 bce,         bcelogits,    ce,  cosineloss,     ctc,         hinge,                   // loss fns
 huber,       kl,           l1,  margin,         mse,         multilabel,
 multimargin, multisoft,    nll, poissonloss,    smoothl1,    softmargin,  triplet,    
 adagrad, adam, adamw, lamb, lbfgs, rmsprop, sgd                                       // optimizers
};

using Args=std::vector<Arg>;        // module's forward arg type(s)

using Attrs=std::tuple<S,           // 0  module symbol
                       Cast,        // 1  module enumeration
                       size_t,      // 2  typeid hash
                       const char*, // 3  description
                       bool,        // 4  true if has non-templatized forward
                       Arg,         // 5  result type
                       size_t,      // 6  min number of arguments
                       size_t,      // 7  max number of arguments
                       Args>;       // 8  argument type(s)

using AttrRef=torch::ArrayRef<Attrs>;    // array reference (for modules/losses)
using ModuleAttrs=std::array<Attrs,128>; // global list of module attributes
using LossAttrs=std::array<Attrs,21>;    // global list of loss modules

enum class Tensormode:char {   // tensor creation modes
 undefined,
 arange, complex, empty,   eye,   full,     linspace, logspace,
 ones,   rand,    randint, randn, randperm, range,    sparse, 
 zeros    
};

enum class Setting:uint8_t {
 undefined,
 addbias,    addzero,            affine,       align,          alloptions,   
 alpha,      amsgrad,            batchfirst,   batchsize,      benchmark,    
 beta,       beta1,              beta2,        bi,             bias,         
 blank,      buffers,            ceiling,      centered,       changetol,    
 channels,   classes,            clipgroup,    clipnorm,       clipvalue,    
 cols,       complexfirst,       countpad,     cuda,           cudadevices,  
 cudnn,      cudnndeterministic, cudnnversion, dampening,      decay,        
 decoder,    decoderlayer,       delta,        detach,         deterministic,
 dictionary, dilate,             dim,          dim0,           dim1,         
 divisor,    dlayers,            droplast,     dropout,        dtype,        
 elayers,    encoder,            encoderlayer, end,            eps,          
 eval,       fn,                 freeze,       full,           globalnorm,   
 gradtol,    groups,             heads,        hidden,         history,      
 ignore,     in,                 in1,          in2,            ind,          
 indices,    init,               inplace,      interopthreads, iter,         
 k,          kdim,               keepdim,      kvbias,         kvzeros,      
 lambda,     lastoffset,         layernorm,    layers,         length,       
 log,        lower,              lr,           lrdecay,        magma,        
 margin,     max,                maxnorm,      mean,           metrics,      
 min,        mkl,                mode,         mps,            momentum,       nesterov,     
 norm,       openmp,             out,          outpad,         outsize,      
 p,          pad,                padflag,      padindex,       padmode,      
 parms,      ratio,              reduce,       rescale,        rows,         
 scale,      search,             shape,        shuffle,        shufflecuda, shuffleseed, size,         
 slope,      smoothing,          sparse,       stackframe,     start,        
 std,        stride,             swap,         sync,           task,
 tasks,      tensor,             threads,      threshold,      track,
 train,      transpose,          trustclip,    trustmax,       trustmin,
 unbiased,   upper,              value,        vdim,           weight,
 zeroinf
};

enum class State:char {
 buffers, depth,     loss,     module, name, options, optimizer,
 parms,   parmgroup, pointer, size,   train, test
};

enum class Attr:char {
 undefined = 0,
 ktype,                                                          // char
 bytes,  densedim, dim, elements,  itemsize, nnz, numel, offset, // long scalars
 ptr, ref, sparsedim, sptr, sref, tensorcount, weakref,
 device, dtype, gradfn, gradient, inputmodule, outputmodule,     // symbol
 layout, memory, result,
 coalesced, contiguous, contiguous2d, contiguous3d, defined,     // boolean
 gradflag, leaf, pinned, sparseflag,
 size, stride,                                                   // long list
 data, storage                                                   // other: list,dict,..
};
 
enum class Metric:char {
 batchloss, loss, accuracy, matches, predict, output, hidden, hiddencell
};
using Metrics = std::vector<Metric>;

enum class Help:char {
 undefined=0, backward, device, dtype, ktype 
};

enum class Enum {  // enums to match pytorch variants
 undefined=-1,
 area,            batchmean,       bicubic,         bilinear,   border,   
 circular,        constant,        conv1d,          conv2d,     conv3d,   
 convtranspose1d, convtranspose2d, convtranspose3d, fanin,      fanout,   
 gelu,            leakyrelu,       linear,          max,        mean,
 mish,            nearest,         nearestexact,    none,       reflect,
 reflection,      relu,            replicate,       same,       sigmoid,
 silu,            sum,             tanh,            trilinear,  valid,
 zeros
};

struct TORCH_API Kmodule;

struct TORCH_API Ktag {
 Class  a = Class::undefined;
 Cast   c = Cast::undefined;
 virtual ~Ktag() = default;

 virtual void set(const Tensor& t) {TORCH_ERROR("unable to set tensor");}
 virtual void set(const TensorVector& v) {TORCH_ERROR("unable to set dict");}
 virtual void set(const TensorDict& d) {TORCH_ERROR("unable to set dictionary");}

 virtual Tensor& tensor() {TORCH_ERROR("unable to retrieve tensor");}
 virtual const Tensor& tensor() const {TORCH_ERROR("unable to retrieve tensor");}
 virtual TensorVector& vector() {TORCH_ERROR("unable to retrieve vector");}
 virtual const TensorVector& vector() const {TORCH_ERROR("unable to retrieve vector");}
 virtual TensorDict& dict() {TORCH_ERROR("unable to retrieve dictionary");}
 virtual const TensorDict& dict() const {TORCH_ERROR("unable to retrieve dictionary");}
 virtual Kmodule* kmodule() {TORCH_ERROR("unable to retrieve module");}
 virtual Module& module() {TORCH_ERROR("unable to retrieve module");}
 virtual const Module& module() const {TORCH_ERROR("unable to retrieve module");}
 virtual Moduleptr& moduleptr() {TORCH_ERROR("unable to retrieve module pointer");}
 virtual const Moduleptr& moduleptr() const {TORCH_ERROR("unable to retrieve module pointer");}
 virtual Optimizer& opt() {TORCH_ERROR("unable to retrieve optimizer");}
 virtual const Optimizer& opt() const {TORCH_ERROR("unable to retrieve optimizer");}
 virtual Optptr& optptr() {TORCH_ERROR("unable to retrieve optimizer pointer");}
 virtual const Optptr& optptr() const {TORCH_ERROR("unable to retrieve optimizer pointer");}
};

struct TORCH_API Kten : public Ktag {
 Tensor t;
 Kten(const Tensor& x) : t(std::move(x)) {a=Class::tensor; c=Cast::tensor;}
 Tensor& tensor() {return t;}
 const Tensor& tensor() const {return t;}
 void set(const Tensor& x) {t=std::move(x);}
};

struct TORCH_API Kvec : public Ktag {
 TensorVector v;
 Kvec(const TensorVector& x) : v(std::move(x)) {a=Class::vector; c=Cast::tensor;}
 TensorVector& vector() {return v;}
 const TensorVector& vector() const {return v;}
 void set(const TensorVector& x) {v=std::move(x);}
};

struct TORCH_API Kdict : public Ktag {
 TensorDict d;
 Kdict(const TensorDict& x,Cast y=Cast::tensor) : d(std::move(x)) {a=Class::dict; c=y;}

 TensorDict& dict() {return d;}
 const TensorDict& dict() const {return d;}
 void set(const TensorDict& x) {d=std::move(x);}
};

struct TORCH_API TrainOptions {
 using Doubles = std::array<double,2>;
 TORCH_ARG(int64_t, batchsize)    = 32;
 TORCH_ARG(int64_t, task)         = 0;
 TORCH_ARG(int64_t, tasks)        = 1;
 TORCH_ARG(int64_t, shuffleseed)  = 0;
 TORCH_ARG(bool,    droplast)     = false;
 TORCH_ARG(bool,    hidden)       = false;
 TORCH_ARG(bool,    shuffle)      = false;
 TORCH_ARG(bool,    shufflecuda)  = false;
 TORCH_ARG(bool,    tensor)       = false;
 TORCH_ARG(bool,    dictionary)   = false;
 TORCH_ARG(bool,    sync)         = false;
 TORCH_ARG(bool,    clipgroup)    = false;
 TORCH_ARG(c10::optional<Doubles>, clipnorm);
 TORCH_ARG(c10::optional<double>,  clipvalue);
 TORCH_ARG(Metrics, metrics) = {Metric::loss};
};

struct TORCH_API TestOptions {
 TORCH_ARG(int64_t, batchsize)  = 100;
 TORCH_ARG(int64_t, task)       = 0;
 TORCH_ARG(int64_t, tasks)      = 1;
 TORCH_ARG(bool,    droplast)   = false;
 TORCH_ARG(bool,    hidden)     = false;
 TORCH_ARG(bool,    tensor)     = false;
 TORCH_ARG(bool,    dictionary) = false;
 TORCH_ARG(Metrics, metrics) = {Metric::loss};
};

struct TORCH_API ForwardOptions {
 TORCH_ARG(Cast,  in) = Cast::undefined;  // type of module accepting input
 TORCH_ARG(Cast, out) = Cast::undefined;  // type of module returning output
 TORCH_ARG(bool,   f) = false;            // true if non-templated forward exists
 TORCH_ARG(Arg,    r) = Arg::undefined;   // type of result
 TORCH_ARG(size_t, n) = 0;                // number of required arguments
 TORCH_ARG(size_t, m) = 0;                // maximum number of arguments
 TORCH_ARG(Args,   a);                    // vector of argument type(s)
};

void forwardoptions(Cast,ForwardOptions&,const Module&); // initializes options during module construction

struct TORCH_API Kmodule : public Ktag {
 Kmodule(Class x,Cast y,const Moduleptr& p) : m(std::move(p)) {a=x; c=y; forwardoptions(c,f,*m);}

 Kmodule* kmodule() {return this;};
 Module& module() {return *m;}
 const Module& module() const {return *m;}
 Moduleptr& moduleptr() {return m;}
 const Moduleptr& moduleptr() const {return m;}

 Moduleptr m;               // generic module pointer, with specific run-tyme type
 ForwardOptions f;          // options describing forward calculation
 c10::optional<Device> d;   // initialized if m.to() called or if forward call uses k array
};

struct TORCH_API Kopt : public Ktag {
 Optptr o;       // shared ptr with optimizer
 Moduleptr m;    // single module or container holding all modules/tensors managed by optimizer
 Kopt(Cast x,const Optptr& y,const Moduleptr& m) : o(std::move(y)),m(std::move(m)) {a=Class::optimizer; c=x;}

 Optimizer& opt() {return *o;}
 const Optimizer& opt() const {return *o;}
 Optptr& optptr() {return o;}
 const Optptr& optptr() const {return o;}
 Module& module() {return *m;}
 const Module& module() const {return *m;}
 Moduleptr& moduleptr() {return m;}
 const Moduleptr& moduleptr() const {return m;}
};

struct TORCH_API Data {
 TORCH_ARG(int64_t, size) = -1;       // size of tensors (along batching dimension)
 TORCH_ARG(int64_t, batchsize) = -1;  // size of batches
 TORCH_ARG(int64_t, batch) = -1;      // current batch
 TORCH_ARG(int64_t, batches) = -1;    // overall number of batches
 public:
 Input x = Empty();                   // model input(s)
 Input y = Empty();                   // model target(s)
 Output z;                            // model output for latest batch
 Tensor l;                            // tensor loss for latest batch (if required)
 Tensor p;                            // permutation index if shuffled
 Generator g;                         // generator (used for permutation index across tasks)
 MetricData m;                        // metrics stored in vector of vectors for each batch
};

struct TORCH_API Kmodel : public Ktag {
 Kmodel(Kmodule *x,Kmodule *y,Kopt *z) : q(*x), l(*y), o(*z) {
  a=Class::model; c=Cast::model;
 }

 Kmodule* kmodule() {return &q;}
 Kmodule* kloss()   {return &l;}
 Kopt*    kopt()    {return &o;}

 Module& module() {return *q.m;}
 const Module& module() const {return *q.m;}
 Moduleptr& moduleptr() {return q.m;}
 const Moduleptr& moduleptr() const {return q.m;}
 Optimizer& opt() {return *o.o;}
 const Optimizer& opt() const {return *o.o;}
 Optptr& optptr() {return o.o;}
 const Optptr& optptr() const {return o.o;}

 Kmodule q;
 Kmodule l;
 Kopt    o;
 TrainOptions train;
 TestOptions  test;
 Data data;
 Data testdata;
};

S krrbuf(const char*);
void dictadd(K,S,K);
void dictadd(K,const char*,K);
bool xind(K,J);
bool xind(K,J,Ktype);
K kptr(void*);
bool ptrtype(K);
bool ptrflag(K);
bool mapped(K);
bool xptr(K);
bool xptr(K,J);
Ktag* xtag(K);
Ktag* xtag(K,J);

bool null(const char*);
bool null(const J);
bool match(const Scalar&,const Scalar&);
K kscalar(const Scalar&);
K resolvedict(K);
K resolve(K);

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
S  inputname(const  Input&);
S outputname(const Output&);

S statekey(State);
J statefind(State,K,bool r=false);
S statesym(State,bool,K,J j=-1);
K statedict(State,K,J j=-1);
K statetable(State,K);

J statedepth(K x,J j=-1);
S statemodule(K x,J j=-1);
S statename(K x,J j=-1);
K stateoptions(K x,J j=-1);
K stateparms(K x,J j=-1);
K statebuffers(K x,J j=-1);
J stategroup(K x,J j=-1);
K statesize(K x,J j=-1);
K statecol(State,K,short t=nh);
void stateparms(S,Module&,K,bool);

S nullsym();
K knull();
bool nullsym(S);
bool nullsym(K);

bool xnull(K);
bool xnull(K,J);
bool xempty(K);
bool xempty(K,J);
bool xarray(K,J);
bool xsym(K);
bool xsym(K,J);
bool xsym(K,S&);
bool xsym(K,J,S&);
bool xsyms(K,S&);
bool xsyms(K,SymArrayRef&);
bool xsyms(K,J,SymArrayRef&);
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
bool xtenarg(K,J,Tensor&,Tensor&);
bool xtenarg(K,J,Tensor&,Tensor&,Tensor&);
bool xtenarg(K,Tensor&,Tensor&);
bool xtenarg(K,Tensor&,Tensor&,Tensor&);
TensorVector xtensors(K x,bool& p,const char* c);

Kmodule* xmodule(K);
Kmodule* xmodule(K,J);
Kmodule* xloss(K);
Kmodule* xloss(K,J);
Kopt* xoptim(K);
Kopt* xoptim(K,J);
Kmodel* xmodel(K);
Kmodel* xmodel(K,J);
bool xparm(K,Cast,S&,Tensor&);
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
S argname(Arg);
Arg argtype(S s,const char *c=nullptr);
K arglist(const Args&);
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
J xdv(K);
J xdv(K,J);
J dvd(K,J);
K dvv(K,J);

c10::optional<Device> firstdevice(const Tensor&);
c10::optional<Device> firstdevice(const TensorVector&);
c10::optional<Device> firstdevice(const TensorDict&);
c10::optional<Device> firstdevice(const Input&);
void sync(int64_t);
void sync(const Device&);

Device defaultdevice(const c10::optional<Device>);
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
bool xfree(K);
void kfree(const std::vector<K>&);
void fn(K,const char*,void*,I);
void randomfn(K);
void mathfn(K);
K attr(K,Ktype,Attr);
S tensortype(Cast);
void castsym(S,Class&,Cast&);
Cast castsym(S);

// tensor & vector routines:
K kget(const Tensor&);
K kget(const LongVector&);
K kget(const DoubleVector&);
K kget(const TensorVector& v,K x=nullptr);
K kget(const TensorDict& d,K x=nullptr);
K kget(const TensorDeque&);
K kget(const Tuple&);
K kget(const Nested&);
K kget(const Input&);
K kget(const Output&);
K kin(const Input&);
K kout(const Output&);
bool broadcast(const Tensor&,const Tensor&);
Tensor kput(K);
Tensor kput(K,J);
TensorDict kputd(K);
TensorVector vec(K,bool b=false);
K kten(const Tensor&);
K kvec(const TensorVector&);
K kdict(const TensorDict&,Cast c=Cast::tensor);
inline K kresult(bool p,const Tensor& t) {return p ? kten(t) : kget(t);}
inline K kresult(bool p,const TensorVector& v) {return p ? kvec(v) : kget(v);}
inline K kresult(bool p,const Tuple& t)  {return kresult(p, TensorVector{std::get<0>(t),std::get<1>(t)});}
inline K kresult(bool p,const Tuple3& t) {return kresult(p, TensorVector{std::get<0>(t),std::get<1>(t),std::get<2>(t)});}
inline K kresult(bool p,const Tuple4& t) {return kresult(p, TensorVector{std::get<0>(t),std::get<1>(t),std::get<2>(t),std::get<3>(t)});}

inline K koutput(TensorVector& v,const Tuple& t) {
 using std::get;
 switch(v.size()) {
  case 0:  v.push_back(get<0>(t)); v.push_back(get<1>(t)); break;
  case 1:  v[0]=get<0>(t); v.push_back(get<1>(t)); break;
  default: v[0]=get<0>(t); v[1]=get<1>(t); break;
 }
 return (K)0;
}

inline K koutput(TensorVector& v,const Tuple3& t) {
 using std::get;
 switch(v.size()) {
  case 0:  v.push_back(get<0>(t)); v.push_back(get<1>(t)); v.push_back(get<2>(t)); break;
  case 1:  v[0]=get<0>(t); v.push_back(get<1>(t)); v.push_back(get<2>(t)); break;
  case 2:  v[0]=get<0>(t); v[1]=get<1>(t); v.push_back(get<2>(t)); break;
  default: v[0]=get<0>(t); v[1]=get<1>(t); v[2]=get<2>(t); break;
 }
 return (K)0;
}

inline K koutput(TensorVector& v,const Tuple4& t) {
 using std::get;
 switch(v.size()) {
  case 0:  v.push_back(get<0>(t)); v.push_back(get<1>(t)); v.push_back(get<2>(t)); v.push_back(get<3>(t)); break;
  case 1:  v[0]=get<0>(t); v.push_back(get<1>(t)); v.push_back(get<2>(t)); v.push_back(get<3>(t)); break;
  case 2:  v[0]=get<0>(t); v[1]=get<1>(t); v.push_back(get<2>(t)); v.push_back(get<3>(t)); break;
  case 3:  v[0]=get<0>(t); v[1]=get<1>(t); v[2]=get<2>(t); v.push_back(get<3>(t)); break;
  default: v[0]=get<0>(t); v[1]=get<1>(t); v[2]=get<2>(t); v[3]=get<3>(t); break;
 }
 return (K)0;
}

K to(Kten*,  const TensorOptions&,bool,bool);
void to(TensorVector&,const TensorOptions&,bool);
void to(TensorDict&,const TensorOptions&,bool);
J tensorlong(const Tensor&,Attr);
S tensorsym(const Tensor&,Attr);
K tensorsize(const Tensor&,Attr);
K tensorattr(const Tensor&,Ktype,Attr);
K vectorattr(const TensorVector&,Ktype,Attr);
K dictattr(const TensorDict&,Ktype,Attr);
K tensorinfo(const Tensor&,bool);
K vectorinfo(const TensorVector&,bool);
void tensorcopy(Tensor&,const Tensor&,bool async=false);
std::vector<int64_t> newsize(const Tensor&,int64_t,int64_t);
int64_t maxsize(const Tensor&,       int64_t d=0);
int64_t maxsize(const TensorVector&, int64_t d=0);
int64_t maxsize(const TensorDict&,   int64_t d=0);
int64_t checksize(const Input&,const Input&);
int64_t fullsize(const Tensor&, int64_t d=0,int64_t n=-1);
int64_t fullsize(const TensorVector&, int64_t d=0,int64_t n=-1);
int64_t fullsize(const TensorDict&,   int64_t d=0,int64_t n=-1);
int64_t batches(int64_t w,int64_t n,bool b=false);
void batch(const Input& x,int64_t i,int64_t w,int64_t d=0,int64_t n=-1);
bool nextbatch(K,int64_t,int64_t);
void batchindex(K,int64_t,int64_t,int64_t);
void setsafe(Tensor& t,int64_t,const IntArrayRef&,const IntArrayRef&);
TensorVector tensorpick(Ktag *,K,bool,Cast,const char*);
void tensorfn(K);


// module routines
using Callbacks=std::array<Moduleptr,3>; // list of callbacks
ModuleAttrs moduleattrs();
Callbacks callbacks();
K kmodule(Cast,const Moduleptr&,Class a=Class::module);
K kmodule(Kmodule*);
void to(Kmodule*,const TensorOptions&,bool);

Moduleptr mcreate(K,J,Cast);
AnyModule anymodule(Cast,const Moduleptr&);

const
c10::optional<std::string>& mname_(const Module&);
c10::optional<std::string>& mname_(Module&);
S mname(const Module&);
std::string mlabel(const char*);
std::string mlabel(const Module&);
std::string mlabel(const Moduleptr&);
std::string mlabel(Kmodule*);
J argstart(K,S);
Cast mcast(const Module&);
Cast mcast(const Moduleptr&);
Cast mcast(const Moduleptr&,bool);
Cast msym(S);
S msym(Cast);
S msym(const Module&);
void msyms(K x,S&,S&);
const Tensor *findtensor(const Module&,const std::string&,Cast);
K modexample(Cast);
K moduleoptions(bool,bool,Cast,const Module&);
K moduleget(bool,bool,const Module&);
Output mforward(Kmodule*,const Input&);
void nnfn(K);

// loss functions:
LossAttrs lossattrs();
Cast lmap(S);
S lmap(Cast);
S lmap(Kmodule*);
Tensor losscalc(Kmodule*,const Tensor&,const Tensor&);
Tensor losscalc(Kmodule*,const Tensor&,const Tensor&,const Tensor&);
Tensor losscalc(Kmodule*,const Tensor&,const Tensor&,const Tensor&,const Tensor&);
Tensor losscalc(Kmodel*,const Output&,const Input&);
K lossexample(Cast);
K lossoptions(bool,Cast,const Module&);
K lossget(bool,bool,Cast,const Module&);
void lossfn(K);

// optimization functions:
S omap(Cast);
size_t osize(const Optimizer&);
J buffersize(Attr,Cast,const Optimizer&);
K kopt(Cast,const Optptr&,const Moduleptr&);
K optget(bool,bool,Cast,const Optimizer&,const Module&);
K optsettings(bool,Cast,const Optimizer&);
K optdefaults(Cast);
void optfn(K);

// model functions:
K modelget(bool,bool,Kmodel*);
Input modelarg(K,J,const char*);
std::tuple<Input,Input> modelargs(K,const char* c);
void modelfn(K);

#ifdef __clang__
# pragma clang diagnostic push
# pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"    // FORWARD_HAS_DEFAULT_ARGS
#endif
//#include "knn.h"
#ifdef __clang__
# pragma clang diagnostic pop
#endif

// global environment
typedef struct Env {
 I cuda;                  // number of CUDA devices
 S nullsym=cs("");        // internal representation of null symbol
 bool frame=false;        // if true, error message returns stack frame
 bool alloptions=true;    // if true, return all option settings, else only non-defaults
 bool complexfirst=true;  // if true, return complex tensor as (real;imag) instead of (real,'imag)

 std::vector<std::tuple<S,Device>> device;

 std::array<std::tuple<Ktype,TypeMeta,char>,8> ktype = {{           //k type -> torch type
  std::make_tuple(KE,     torch::scalarTypeToTypeMeta(torch::kFloat),   'e'),
  std::make_tuple(KF,     torch::scalarTypeToTypeMeta(torch::kDouble),  'f'),
  std::make_tuple(KJ,     torch::scalarTypeToTypeMeta(torch::kLong),    'j'),
  std::make_tuple(KI,     torch::scalarTypeToTypeMeta(torch::kInt),     'i'),
  std::make_tuple(KSHORT, torch::scalarTypeToTypeMeta(torch::kShort),   'h'),
  std::make_tuple(KB,     torch::scalarTypeToTypeMeta(torch::kBool),    'b'),
  std::make_tuple(KG,     torch::scalarTypeToTypeMeta(torch::kByte),    'x'),
  std::make_tuple(KC,     torch::scalarTypeToTypeMeta(torch::kChar),    'c')
 }};

 std::array<std::tuple<S,TypeMeta,Ktype,char>,12> dtype = {{       //sym -> torch type -> k type
  std::make_tuple(cs("float"),   torch::scalarTypeToTypeMeta(torch::kFloat),         KE,     'e'),
  std::make_tuple(cs("double"),  torch::scalarTypeToTypeMeta(torch::kDouble),        KF,     'f'),
  std::make_tuple(cs("half"),    torch::scalarTypeToTypeMeta(torch::kHalf),          KE,     'e'),
  std::make_tuple(cs("bool"),    torch::scalarTypeToTypeMeta(torch::kBool),          KB,     'b'),
  std::make_tuple(cs("byte"),    torch::scalarTypeToTypeMeta(torch::kByte),          KG,     'x'),
  std::make_tuple(cs("char"),    torch::scalarTypeToTypeMeta(torch::kChar),          KC,     'c'),
  std::make_tuple(cs("long"),    torch::scalarTypeToTypeMeta(torch::kLong),          KJ,     'j'),
  std::make_tuple(cs("int"),     torch::scalarTypeToTypeMeta(torch::kInt),           KI,     'i'),
  std::make_tuple(cs("short"),   torch::scalarTypeToTypeMeta(torch::kShort),         KSHORT, 'h'),
  std::make_tuple(cs("chalf"),   torch::scalarTypeToTypeMeta(torch::kComplexHalf),   KE,     'e'),
  std::make_tuple(cs("cfloat"),  torch::scalarTypeToTypeMeta(torch::kComplexFloat),  KE,     'e'),
  std::make_tuple(cs("cdouble"), torch::scalarTypeToTypeMeta(torch::kComplexDouble), KF,     'f')
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

 std::array<std::tuple<S,Class>,9> kclass = {{         //higher level object names
  std::make_tuple(cs("tensor"),     Class::tensor),          
  std::make_tuple(cs("vector"),     Class::vector),
  std::make_tuple(cs("dictionary"), Class::dict),
  std::make_tuple(cs("module"),     Class::module),
  std::make_tuple(cs("loss"),       Class::loss),
  std::make_tuple(cs("optimizer"),  Class::optimizer),
  std::make_tuple(cs("model"),      Class::model),
  std::make_tuple(cs("train"),      Class::train),
  std::make_tuple(cs("test"),       Class::test)
 }};

 std::array<std::tuple<S,Arg>,6> arg = {{               // type of inputs & output for modules
  std::make_tuple(cs("bool"),       Arg::boolean),      // accomodate bool arg of MultiHeadAttention
  std::make_tuple(cs("tensor"),     Arg::tensor),          
  std::make_tuple(cs("tuple"),      Arg::tuple),          
  std::make_tuple(cs("nested"),     Arg::nested),          
  std::make_tuple(cs("vector"),     Arg::vector),
  std::make_tuple(cs("dictionary"), Arg::dict)
 }};

 std::array<std::tuple<S,Cast>,3> tensortype = {{       //distiguish tensor from parameter & buffer
  std::make_tuple(cs("tensor"),     Cast::tensor),          
  std::make_tuple(cs("parameter"),  Cast::parameter),          
  std::make_tuple(cs("buffer"),     Cast::buffer)          
 }};

 std::array<S,std::variant_size_v<Input>> in = {{
  cs("tensor"),
  cs("vector"),
  cs("dictionary"),
  cs("empty")
 }};

 std::array<S,std::variant_size_v<Output>> out = {{
  cs("tensor"),
  cs("tuple"),
  cs("nested"),
  cs("vector")
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

 ModuleAttrs modules=moduleattrs();
 LossAttrs   loss=lossattrs();
 Callbacks cb=callbacks();

 std::array<std::tuple<S,Setting>,16> cset = {{            // configuration settings
 std::make_tuple(cs("mkl"),                Setting::mkl),
 std::make_tuple(cs("openmp"),             Setting::openmp),
 std::make_tuple(cs("threads"),            Setting::threads),
 std::make_tuple(cs("interopthreads"),     Setting::interopthreads),
 std::make_tuple(cs("mps"),                Setting::mps),
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

 std::array<std::tuple<S,Setting>,89> mset = {{        // module option sym -> enum
  std::make_tuple(cs("addbias"),      Setting::addbias),
  std::make_tuple(cs("addzero"),      Setting::addzero),
  std::make_tuple(cs("affine"),       Setting::affine),
  std::make_tuple(cs("alpha"),        Setting::alpha),
  std::make_tuple(cs("align"),        Setting::align),
  std::make_tuple(cs("batchfirst"),   Setting::batchfirst),
  std::make_tuple(cs("beta"),         Setting::beta),
  std::make_tuple(cs("bi"),           Setting::bi),
  std::make_tuple(cs("bias"),         Setting::bias),
  std::make_tuple(cs("buffers"),      Setting::buffers),
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
  std::make_tuple(cs("dim0"),         Setting::dim0),
  std::make_tuple(cs("dim1"),         Setting::dim1),
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
  std::make_tuple(cs("length"),       Setting::length),
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
  std::make_tuple(cs("parms"),        Setting::parms),
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

// training options, 1st flag true if training option, 2nd flag true if an evaluation option
 std::array<std::tuple<S,Setting,bool,bool>,15> train = {{
 std::make_tuple(cs("batchsize"),     Setting::batchsize,   true,  true),
 std::make_tuple(cs("clipgroup"),     Setting::clipgroup,   true,  false),
 std::make_tuple(cs("clipnorm"),      Setting::clipnorm,    true,  false),
 std::make_tuple(cs("clipvalue"),     Setting::clipvalue,   true,  false),
 std::make_tuple(cs("dictionary"),    Setting::dictionary,  true,  true),
 std::make_tuple(cs("droplast"),      Setting::droplast,    true,  true),
 std::make_tuple(cs("hidden"),        Setting::hidden,      true,  true),
 std::make_tuple(cs("metrics"),       Setting::metrics,     true,  true),
 std::make_tuple(cs("shuffle"),       Setting::shuffle,     true,  false),
 std::make_tuple(cs("shufflecuda"),   Setting::shufflecuda, true,  false),
 std::make_tuple(cs("shuffleseed"),   Setting::shuffleseed, true,  false),
 std::make_tuple(cs("sync"),          Setting::sync,        true,  false),
 std::make_tuple(cs("task"),          Setting::task,        true,  true),
 std::make_tuple(cs("tasks"),         Setting::tasks,       true,  true),
 std::make_tuple(cs("tensor"),        Setting::tensor,      true,  true)
}};

// module state dictionary keys: map symbol -> enum
 std::array<std::tuple<S,State>,13> state = {{  
  std::make_tuple(cs("buffers"),   State::buffers),
  std::make_tuple(cs("depth"),     State::depth),
  std::make_tuple(cs("loss"),      State::loss),
  std::make_tuple(cs("module"),    State::module),
  std::make_tuple(cs("name"),      State::name),
  std::make_tuple(cs("options"),   State::options),
  std::make_tuple(cs("optimizer"), State::optimizer),
  std::make_tuple(cs("parms"),     State::parms),
  std::make_tuple(cs("parmgroup"), State::parmgroup),
  std::make_tuple(cs("pointer"),   State::pointer),
  std::make_tuple(cs("size"),      State::size),
  std::make_tuple(cs("test"),      State::test),
  std::make_tuple(cs("train"),     State::train)
 }};

 std::array<std::tuple<S,Setting>,14> lset = {{     // loss option sym -> enum
  std::make_tuple(cs("beta"),      Setting::beta),
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

 std::array<std::tuple<S,Cast,std::string>,7> opt = {{        //optimizer: map symbol -> enum
  std::make_tuple(cs("adagrad"), Cast::adagrad, "torch.optim.Adagrad"),
  std::make_tuple(cs("adam"),    Cast::adam,    "torch.optim.Adam"),
  std::make_tuple(cs("adamw"),   Cast::adamw,   "torch.optim.AdamW"),
  std::make_tuple(cs("lamb"),    Cast::lamb,    ""),
  std::make_tuple(cs("lbfgs"),   Cast::lbfgs,   "torch.optim.LBFGS"),
  std::make_tuple(cs("rmsprop"), Cast::rmsprop, "torch.optim.RMSprop"),
  std::make_tuple(cs("sgd"),     Cast::sgd,     "torch.optim.SGD")
 }};

 std::array<std::tuple<S,Setting>,24> oset = {{         //optimizer setting: map symbol -> enum
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
  std::make_tuple(cs("globalnorm"), Setting::globalnorm),
  std::make_tuple(cs("gradtol"),    Setting::gradtol),
  std::make_tuple(cs("history"),    Setting::history),
  std::make_tuple(cs("init"),       Setting::init),
  std::make_tuple(cs("iter"),       Setting::iter),
  std::make_tuple(cs("lrdecay"),    Setting::lrdecay),
  std::make_tuple(cs("lr"),         Setting::lr),
  std::make_tuple(cs("momentum"),   Setting::momentum),
  std::make_tuple(cs("nesterov"),   Setting::nesterov),
  std::make_tuple(cs("search"),     Setting::search),
  std::make_tuple(cs("trustclip"),  Setting::trustclip),
  std::make_tuple(cs("trustmax"),   Setting::trustmax),
  std::make_tuple(cs("trustmin"),   Setting::trustmin),
  std::make_tuple(cs("unbiased"),   Setting::unbiased)
 }};

 std::array<std::tuple<S,Attr>,38> attr = {{            //attributes: map symbol -> enum
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
  std::make_tuple(cs("inputmodule"),  Attr::inputmodule),
  std::make_tuple(cs("itemsize"),     Attr::itemsize),
  std::make_tuple(cs("ktype"),        Attr::ktype),
  std::make_tuple(cs("layout"),       Attr::layout),
  std::make_tuple(cs("leaf"),         Attr::leaf),
  std::make_tuple(cs("memory"),       Attr::memory),
  std::make_tuple(cs("nnz"),          Attr::nnz),
  std::make_tuple(cs("numel"),        Attr::numel),
  std::make_tuple(cs("offset"),       Attr::offset),
  std::make_tuple(cs("outputmodule"), Attr::outputmodule),
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

 std::array<std::tuple<S,bool,bool,const char *>,4> backsym = {{     //map sym to booleans for retain_graph & create_graph
  std::make_tuple(cs("free"),       false, false, "free graph, no higher order derivatives"),
  std::make_tuple(cs("retain"),     true,  false, "retain graph, no higher order derivatives"),
  std::make_tuple(cs("create"),     true,  true,  "retain graph, create graph for higher order derivatives"),
  std::make_tuple(cs("createfree"), false, true,  "free graph & create graph for higher order derivatives (unused?)")
 }};

 std::array<std::tuple<S,Metric>,8> metric = {{
  std::make_tuple(cs("batchloss"),  Metric::batchloss),
  std::make_tuple(cs("loss"),       Metric::loss),
  std::make_tuple(cs("accuracy"),   Metric::accuracy),
  std::make_tuple(cs("matches"),    Metric::matches),
  std::make_tuple(cs("predict"),    Metric::predict),
  std::make_tuple(cs("output"),     Metric::output),
  std::make_tuple(cs("hidden"),     Metric::hidden),
  std::make_tuple(cs("hiddencell"), Metric::hiddencell)
 }};

 std::array<std::tuple<S,Help>,4> helptopic = {{
  std::make_tuple(cs("backward"),  Help::backward),
  std::make_tuple(cs("device"),    Help::device),
  std::make_tuple(cs("dtype"),     Help::dtype),
  std::make_tuple(cs("ktype"),     Help::ktype),
 }};

 // array must match order of Enum, so enum can be used as index
 std::array<std::tuple<S,Enum>,36> enums = {{
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
  std::make_tuple(cs("nearestexact"),    Enum::nearestexact),
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
 S operator()(const torch::enumtype::kNearestExact&)     const { return std::get<0>(env().enums[(size_t)Enum::nearestexact]);}
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
