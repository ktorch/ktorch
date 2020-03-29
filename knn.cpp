#include "ktorch.h"
#include "knn.h"

// append a module option to a k dictionary given dict,name & value
#define OPTION(x,k,v) dictadd(x, mset(Setting::k), v)

// append a module with name if not null (method needs `std::string` ??)
#define PUSH(q,n,m) n ? q->push_back(std::string(n),m) : q->push_back(m)

// ----------------------------------------------------------------------------
// kseq - allocate an object to store a pointer to a sequential module
// seqto - given sequential module & options, change device/data type
// ----------------------------------------------------------------------------
K kmodule(Cast c,const AnyModule& m) {return kptr(new Kmodule(Class::module,c,m));}

K kseq(const Sequential& q) {return kptr(new Kseq(q));}

K seqto(Kseq* q,const TensorOptions& o,bool a) {
 auto s=torch::typeMetaToScalarType(o.dtype());
 if(o.has_device() && o.has_dtype()) q->q->to(o.device(),s,a);
 else if(o.has_device())             q->q->to(o.device(),a);
 else                                q->q->to(s,a);
 return (K)0;
}

// --------------------------------------------------------------------------------------------
// enum<-rnnfn(sym)    match symbol to enum for activation function
// sym<-rnnfn(options) return symbol matching activation fn, else null (e.g. for gru/lstm)
// rnnfn(options,sym)  set activation function if rnn options, else no-op
// --------------------------------------------------------------------------------------------
/*
static torch::nn::RNNActivation rnnfn(S s) {
 for(auto& m:env().rnnfn) if (s==std::get<0>(m)) return std::get<1>(m);
 AT_ERROR("Unrecognized rnn activiation function: ",s);
}

template<typename O> static S rnnfn(O& o) {return nullptr;}
template<> S rnnfn<torch::nn::RNNOptions>(torch::nn::RNNOptions& o) {
 for(auto& m:env().rnnfn) if (o.activation()==std::get<1>(m)) return std::get<0>(m);
 AT_ERROR("Unrecognized rnn activiation function: ",(I)o.activation());
}

template<typename O> static void rnnfn(O& o,torch::nn::RNNActivation f) {}
template<> void rnnfn<torch::nn::RNNOptions>(torch::nn::RNNOptions& o,torch::nn::RNNActivation f) {o.activation(f);}
*/

// -----------------------------------------------------------------------------------
// msym - map to/from sym & enum for module, e.g. `conv3d <-> Cast::conv3d
// mset - map to/from sym & enum for module options, e.g. `bias <-> Setting::bias
// container - given module/module cast, return true if container module
// -----------------------------------------------------------------------------------
static S msym(Cast c) {
 for(auto& m:env().module) if(c==std::get<1>(m)) return std::get<0>(m);
 AT_ERROR("Unrecognized module: ",(I)c);
}

static Cast msym(S s) {
 for(auto& m:env().module) if(s==std::get<0>(m)) return std::get<1>(m);
 AT_ERROR("Unrecognized module: ",s);
}

static S mset(Setting o) {
 for(auto& m:env().mset) if(o==std::get<1>(m)) return std::get<0>(m);
 AT_ERROR("Unrecognized module option: ",(I)o);
}

static Setting mset(S s) {
 for(auto& m:env().mset) if(s==std::get<0>(m)) return std::get<1>(m);
 AT_ERROR("Unrecognized option: ",s);
}

bool container(Cast c) {
 switch(c) {
  case Cast::sequential:
  case Cast::join:
   return true;
  default: return false;
 }
}

bool container(const Module& m) {
 if       (m.as<torch::nn::Sequential>()) { return true;
 } else if(m.as<Join>())                  { return true;
 } else                                   { return false;
 }
}

// ------------------------------------------------------------------------------------------
// mkeys - keys for dict/table of module state: `depth`module`name`options`parms`buffers
// 
// ------------------------------------------------------------------------------------------
K mkeys(bool b) {
 K x=ktn(KS, b ? 6 : 4); J i=0;
 for(auto& m:env().mstate) {
  kS(x)[i++]=std::get<0>(m);
  if(i==x->n) break;
 }
 return x;
}

// ----------------------------------------------------------------------------------------------------
// covers of input checking fns with error msg specific to module settings and module names:
// ----------------------------------------------------------------------------------------------------
// mbool - check positional args or name-value pairs for boolean, else error w'module & option name
// mode  - check positional args of name-value pairs for symbol,  else error w'module & option name
// int64 - check positional args or name-value pairs for long int, else error w'module & option
// int64n - int64 but returns optional, i.e. nullopt if k value is null
// mdouble - check for double(or long) from positional or name-value pair arg
// optdouble - call mdouble() but return null if k null supplied
// ----------------------------------------------------------------------------------------------------
static bool mbool(K x,J i,Cast c,Setting s) {
 bool b;
 TORCH_CHECK(xbool(x,i,b), msym(c)," ",mset(s),": expected boolean scalar, given ",kname(x,i));
 return b;
}

static bool mbool(const Pairs& p,Cast c) {
 TORCH_CHECK(p.t==-KB, msym(c)," ",p.k,": expected boolean scalar, given ",kname(p.t));
 return p.b;
}

static S mode(K x,J i,Cast c,Setting s) {
 S m;
 TORCH_CHECK(xsym(x,i,m), msym(c)," ",mset(s),": expected symbol, given ",kname(x,i));
 return m;
}

static S mode(const Pairs& p,Cast c) {
 TORCH_CHECK(p.t==-KS, msym(c)," ",p.k,": expected symbol, given ",kname(p.t));
 return p.s;
}

static int64_t int64(K x,J i,Cast c,Setting s) {
 int64_t n;
 TORCH_CHECK(xint64(x,i,n), msym(c)," ",mset(s),": expected long scalar, given ",kname(x,i));
 return n;
}

static int64_t int64(const Pairs& p,Cast c) {
 TORCH_CHECK(p.t==-KJ, msym(c)," ",p.k,": expected long scalar, given ",kname(p.t));
 return p.j;
}

static c10::optional<int64_t> int64n(K x,J i,Cast c,Setting s) {auto n=int64(x,i,c,s); if(n==nj) return c10::nullopt; else return n;}
static c10::optional<int64_t> int64n(const Pairs& p,Cast c)    {auto n=int64(p,c);     if(n==nj) return c10::nullopt; else return n;}

static double mdouble(K x,J i,Cast c,Setting s) {
 double f;
 TORCH_CHECK(xnum(x,i,f), msym(c)," ",mset(s),": expected double, given ",kname(x,i));
 return f;
}

static double mdouble(const Pairs& p,Cast c) {
 TORCH_CHECK(p.t==-KJ || p.t==-KF, msym(c)," ",p.k,": expected double, given ",kname(p.t));
 return pdouble(p);
}

static c10::optional<double> optdouble(K x,J i,Cast c,Setting s) {double d=mdouble(x,i,c,s); if(d==d) return d; else return c10::nullopt;}
static c10::optional<double> optdouble(const Pairs& p,Cast c)    {double d=mdouble(p,c);     if(d==d) return d; else return c10::nullopt;}

// -------------------------------------------------------------------------------------------------
// exarray - check positional or name-value args for long(s), return expanding array,  else error
// exoptional - similar to exarray, for optional long(s), return expanding array with nulls
// exdouble - similar to exarray, but for double array
// -------------------------------------------------------------------------------------------------
template<size_t D> static ExpandingArray<D> exarray(K a,J i,Cast c,Setting s) {
 K x=kK(a)[i];
 TORCH_CHECK(x->t==-KJ || x->t==KJ, msym(c)," ",mset(s),": expected long(s), given ",kname(x->t));
 TORCH_CHECK(x->t==-KJ || x->n==D,  msym(c)," ",mset(s),": expected scalar or ",D,"-element input, given ",x->n,"-element list");
 if(x->t==-KJ)
  return ExpandingArray<D>(x->j);
 else
  return ExpandingArray<D>(IntArrayRef((int64_t*)kJ(x),x->n));
}

template<size_t D> static ExpandingArray<D> exarray(const Pairs& p,Cast c) {
 TORCH_CHECK(p.t==-KJ || p.t==KJ,   msym(c)," ",p.k,": expected long(s), given ",kname(p.t));
 TORCH_CHECK(p.t==-KJ || p.v->n==D, msym(c)," ",p.k,": expected scalar or ",D,"-element input, given ",p.v->n,"-element list");
 if(p.t==-KJ)
  return ExpandingArray<D>(p.j);
 else
  return ExpandingArray<D>(IntArrayRef((int64_t*)kJ(p.v),p.v->n));
}

template<size_t D> static Exoptional<D> exoptional(J j) {
 return (j == nj) ? Exoptional<D>(c10::nullopt) : Exoptional<D>(j);
}

template<size_t D> static Exoptional<D> exoptional(K x) {
 auto a=Exoptional<D>(IntArrayRef((int64_t*)kJ(x),x->n));
 for(J i=0;i<x->n;++i) if((*a)[i].value() == nj) (*a)[i]=c10::nullopt;
 return a;
}

template<size_t D> static Exoptional<D> exoptional(K a,J i,Cast c,Setting s) {
 K x=kK(a)[i];
 TORCH_CHECK(x->t==-KJ || x->t==KJ, msym(c)," ",mset(s),": expected long(s), given ",kname(x->t));
 TORCH_CHECK(x->t==-KJ || x->n==D,  msym(c)," ",mset(s),": expected scalar or ",D,"-element input, given ",x->n,"-element list");
 return x->t == -KJ ? exoptional<D>(x->j) : exoptional<D>(x);
}

template<size_t D> static Exoptional<D> exoptional(const Pairs& p,Cast c) {
 TORCH_CHECK(p.t==-KJ || p.t==KJ,   msym(c)," ",p.k,": expected long(s), given ",kname(p.t));
 TORCH_CHECK(p.t==-KJ || p.v->n==D, msym(c)," ",p.k,": expected scalar or ",D,"-element input, given ",p.v->n,"-element list");
 return p.t == -KJ ? exoptional<D>(p.j) : exoptional<D>(p.v);
}

template<size_t D> static Exdouble<D> exdouble(K a,J i,Cast c,Setting s) {
 K x=kK(a)[i];
 TORCH_CHECK(x->t==-KF || x->t==KF, msym(c)," ",mset(s),": expected double(s), given ",kname(x->t));
 TORCH_CHECK(x->t==-KF || x->n==D,  msym(c)," ",mset(s),": expected scalar or ",D,"-element input, given ",x->n,"-element list");
 if(x->t==-KF)
  return Exdouble<D>(x->f);
 else
  return Exdouble<D>(torch::ArrayRef<double>(kF(x),x->n));
}

template<size_t D> static Exdouble<D> exdouble(const Pairs& p,Cast c) {
 TORCH_CHECK(p.t==-KF || p.t==KF,   msym(c)," ",p.k,": expected double(s), given ",kname(p.t));
 TORCH_CHECK(p.t==-KF || p.v->n==D, msym(c)," ",p.k,": expected scalar or ",D,"-element input, given ",p.v->n,"-element list");
 if(p.t==-KF)
  return Exdouble<D>(p.f);
 else
  return Exdouble<D>(torch::ArrayRef<double>(kF(p.v),p.v->n));
}

// --------------------------------------------------------------------------------------
// batchnorm - get/set batch norm & instance norm options
//             both module types and dimensions(1,2,3d) use the same options structure
//             except batch norm's momentum is an optional double
// --------------------------------------------------------------------------------------
static double momentum(c10::optional<double> x) {return x.value();}

template<typename O> static O batchnorm(K x,J i,Cast c) {
 O o(0);
 bool in=false; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
   switch(j) {
    case 0: o.num_features(int64(x,i+j,c,Setting::in)); in=true; break;
    case 1: o.eps(mdouble(x,i+j,c,Setting::eps));break;
    case 2: o.momentum(mdouble(x,i+j,c,Setting::momentum)); break;
    case 3: o.affine(mbool(x,i+j,c,Setting::affine)); break;
    case 4: o.track_running_stats(mbool(x,i+j,c,Setting::affine)); break;
    default: AT_ERROR(msym(c),": up to 5 positional arguments expected, ",n," given");
  }
 while(xpair(p))
  switch(mset(p.k)) {
   case Setting::in:       o.num_features(int64(p,c)); break;
   case Setting::eps:      o.eps(mdouble(p,c)); break;
   case Setting::momentum: o.momentum(mdouble(p,c)); break;
   case Setting::affine:   o.affine(mbool(p,c)); break;
   case Setting::track:    o.track_running_stats(mbool(p,c)); break;
   default: AT_ERROR("Unrecognized ",msym(c)," option: ",p.k); break;
  }
 TORCH_CHECK(in,msym(c),": number of input features not defined");
 return o;
}

template<typename O> static void batchnorm(bool a,K x,const O& o) {
 O d(o.num_features());
 OPTION(x, in, kj(o.num_features()));
 if(a || (o.eps()      != d.eps()))      OPTION(x, eps,       kf(o.eps()));
 if(a || (o.momentum() != d.momentum())) OPTION(x, momentum,  kf(momentum(o.momentum())));
 if(a || (o.affine()   != d.affine()))   OPTION(x, affine,    kb(o.affine()));
 if(a || (o.track_running_stats() != d.track_running_stats())) OPTION(x, track, kb(o.track_running_stats()));
}

// -------------------------------------------------------------------------------------
// localnorm - local response norm, cross map 2d norm, get/set options size,alpha,beta,k
// -------------------------------------------------------------------------------------
template<typename O> static O localnorm(K x,J i,Cast c) {
 O o(0);
 bool b=c==Cast::localnorm,sz=false; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
   switch(j) {
    case 0: o.size(int64(x,i+j,c,Setting::size)); sz=true; break;
    case 1: o.alpha(mdouble(x,i+j,c,Setting::alpha)); break;
    case 2: o.beta(mdouble(x,i+j,c,Setting::beta)); break;
    case 3: b ? o.k(mdouble(x,i+j,c,Setting::k)) : o.k(int64(x,i+j,c,Setting::k)); break;
    default: AT_ERROR(msym(c),": up to 4 positional arguments expected, ",n," given");
  }
 while(xpair(p))
  switch(mset(p.k)) {
   case Setting::size:  o.size(int64(p,c)); sz=true; break;
   case Setting::alpha: o.alpha(mdouble(p,c)); break;
   case Setting::beta:  o.beta(mdouble(p,c)); break;
   case Setting::k:     b ? o.k(mdouble(p,c)) : o.k(int64(p,c)); break;
   default: AT_ERROR("Unrecognized ",msym(c)," option: ",p.k); break;
  }
 TORCH_CHECK(sz, msym(c),": specify no. of neighboring channels to use for normalization");
 return o;
}

template<typename O> static void localnorm(bool a,K x,Cast c,const O& o) {
 O d(o.size());
 OPTION(x, size, kj(o.size()));
 if(a || (o.alpha() != d.alpha())) OPTION(x, alpha, kf(o.alpha()));
 if(a || (o.beta()  != d.beta()))  OPTION(x, beta,  kf(o.beta()));
 if(a || (o.k()     != d.k()))     OPTION(x, k,     c==Cast::localnorm ? kf(o.k()) : kj(o.k()));
}

// --------------------------------------------------------------------------------------
// groupnorm - group norm, get/set number of groups,channels,eps,affine flag
// --------------------------------------------------------------------------------------
static torch::nn::GroupNormOptions groupnorm(K x,J i,Cast c) {
 torch::nn::GroupNormOptions o(0,0);
 bool g=false,h=false; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
   switch(j) {
    case 0: o.num_groups(int64(x,i+j,c,Setting::groups)); g=true; break;
    case 1: o.num_channels(int64(x,i+j,c,Setting::channels)); h=true; break;
    case 2: o.eps(mdouble(x,i+j,c,Setting::eps)); break;
    case 3: o.affine(mbool(x,i+j,c,Setting::affine)); break;
    default: AT_ERROR(msym(c),": up to 4 positional arguments expected, ",n," given");
  }
 while(xpair(p))
  switch(mset(p.k)) {
   case Setting::groups:   o.num_groups(int64(p,c)); g=true; break;
   case Setting::channels: o.num_channels(int64(p,c)); h=true; break;
   case Setting::eps:      o.eps(mdouble(p,c)); break;
   case Setting::affine:   o.affine(mbool(p,c)); break;
   default: AT_ERROR("Unrecognized ",msym(c)," option: ",p.k); break;
  }
 TORCH_CHECK(g, msym(c),": specify no. of groups to separate the channels into");
 TORCH_CHECK(h, msym(c),": specify no. of channels expected in input");
 return o;
}

static void groupnorm(bool a,K x,const torch::nn::GroupNormOptions& o) {
 torch::nn::GroupNormOptions d(o.num_groups(),o.num_channels());
 OPTION(x, groups,   kj(o.num_groups()));
 OPTION(x, channels, kj(o.num_channels()));
 if(a || (o.eps()    != d.eps()))    OPTION(x, eps,    kf(o.eps()));
 if(a || (o.affine() != d.affine())) OPTION(x, affine, kb(o.affine()));
}

// --------------------------------------------------------------------------------------
// layernorm - get/set shape,eps,affine flag for layer normalization
// --------------------------------------------------------------------------------------
static torch::nn::LayerNormOptions layernorm(K x,J i,Cast c) {
 torch::nn::LayerNormOptions o({}); IntArrayRef a; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
   switch(j) {
    case 0: TORCH_CHECK(xsize(x,i+j,a), msym(c),": expecting 1st arg of normalized shape(s)"); break;
    case 1: o.eps(mdouble(x,i+j,c,Setting::eps)); break;
    case 2: o.elementwise_affine(mbool(x,i+j,c,Setting::affine)); break;
    default: AT_ERROR(msym(c),": up to 3 positional arguments expected, ",n," given");
  }
 while(xpair(p))
  switch(mset(p.k)) {
   case Setting::shape:  psize(p,a); break;
   case Setting::eps:    o.eps(mdouble(p,c)); break;
   case Setting::affine: o.elementwise_affine(mbool(p,c)); break;
   default: AT_ERROR("Unrecognized ",msym(c)," option: ",p.k); break;
  }
 TORCH_CHECK(a.size(), msym(c),": no normalized shape given");
 return o.normalized_shape(a.vec());
}

static void layernorm(bool a,K x,const torch::nn::LayerNormOptions& o) {
 torch::nn::LayerNormOptions d(o.normalized_shape());
 OPTION(x, shape, klist(o.normalized_shape().size(),o.normalized_shape().data()));
 if(a || (o.eps()    != d.eps())) OPTION(x, eps, kf(o.eps()));
 if(a || (o.elementwise_affine() != d.elementwise_affine())) OPTION(x, affine, kb(o.elementwise_affine()));
}

// --------------------------------------------------------------------------------------
// normalize - pytorch has functional form only
// --------------------------------------------------------------------------------------
static torch::nn::functional::NormalizeFuncOptions normalize(K x,J i,Cast c,Tensor& r) {
 Pairs p; J n=xargc(x,i,p); torch::nn::functional::NormalizeFuncOptions o;
 if(n>0 && xten(x,i+n-1,r)) n--;
 for(J j=0;j<n;++j)
  switch(j) {
   case 0: o.p(mdouble(x,i+j,c,Setting::p)); break;
   case 1: o.dim(int64(x,i+j,c,Setting::dim)); break;
   case 2: o.eps(mdouble(x,i+j,c,Setting::eps)); break;
   default: AT_ERROR(msym(c),": unrecognized positional arg(s), up to 4 args(p,dim,eps,output tensor) expected, ",n," supplied");
  }
 while(xpair(p))
  switch(mset(p.k)) {
   case Setting::p: o.p(mdouble(p,c)); break;
   case Setting::dim: o.dim(int64(p,c)); break;
   case Setting::eps: o.eps(mdouble(p,c)); break;
   case Setting::out: if(!pempty(p)) pten(p,r);
   default: AT_ERROR("Unrecognized option: ",p.k," for normalize");
  }
 if(r.defined()) 
  o.out(r);
 return o;
}

KAPI Normalize(K x) {
 KTRY
  namespace f=torch::nn::functional;
  Tensor r,*t=nullptr;
  if(x->t || (t=xten(x))) {
   return kresult(t, f::normalize(t ? *t : kput(x), f::NormalizeFuncOptions()));
  } else {
   t=xten(x,0);
   return kresult(t||r.defined(), f::normalize(t ? *t : kput(x,0), normalize(x,1,Cast::normalize,r)));
  }
 KCATCH("normalize");
}

// --------------------------------------------------------------------------------------
// convpad - translate symbol to variant used for padding mode
// conv - create 1-3d convolution, set dictionary given module
//        with version 1.4, the c++ ConvImpl class was split into regular & transposed
//        ConvOptions & ConvTransOptions have different members, 
// convtran - similar to conv() except adds output_padding and changes position order
// --------------------------------------------------------------------------------------
static torch::nn::detail::conv_padding_mode_t convpad(S s) {
 switch(emap(s)) {
  case Enum::zeros:    return torch::kZeros;
  case Enum::circular: return torch::kCircular;
  default: AT_ERROR("unrecognized padding mode: ",s); break;
 }
}

template<size_t D> static torch::nn::ConvOptions<D> conv(K x,J i,Cast c) {
 torch::nn::ConvOptions<D> o(0,0,0);
 bool in=false,out=false,sz=false; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
   switch(j) {
    case 0: o.in_channels (int64(x,i+j,c,Setting::in));        in=true; break;
    case 1: o.out_channels(int64(x,i+j,c,Setting::in));       out=true; break;
    case 2: o.kernel_size (exarray<D>(x,i+j,c,Setting::size)); sz=true; break;
    case 3: o.stride      (exarray<D>(x,i+j,c,Setting::stride));   break;
    case 4: o.padding     (exarray<D>(x,i+j,c,Setting::pad));      break;
    case 5: o.dilation    (exarray<D>(x,i+j,c,Setting::dilate));   break;
    case 6: o.groups      (int64(x,i+j,c,Setting::groups));        break;
    case 7: o.bias        (mbool    (x,i+j,c,Setting::bias));      break;
    case 8: o.padding_mode(convpad(mode(x,i+j,c,Setting::padmode))); break;
    default: AT_ERROR(msym(c),": up to 9 positional arguments expected, ",n," given");
  }
 while(xpair(p))
  switch(mset(p.k)) {
   case Setting::in:        o.in_channels (int64(p,c));     in=true; break;
   case Setting::out:       o.out_channels(int64(p,c));    out=true; break;
   case Setting::size:      o.kernel_size (exarray<D>(p,c)); sz=true; break;
   case Setting::stride:    o.stride      (exarray<D>(p,c)); break;
   case Setting::pad:       o.padding     (exarray<D>(p,c)); break;
   case Setting::dilate:    o.dilation    (exarray<D>(p,c)); break;
   case Setting::groups:    o.groups      (int64(p,c));     break;
   case Setting::bias:      o.bias        (mbool(p,c));     break;
   case Setting::padmode:   o.padding_mode(convpad(mode(p,c)));   break;
   default: AT_ERROR("Unrecognized convolution option: ",p.k); break;
  }
 TORCH_CHECK(in,  msym(c),": number of input channels not defined");
 TORCH_CHECK(out, msym(c),": number of output channels not defined");
 TORCH_CHECK(sz,  msym(c),": no kernel size(s) given");
 return o;
}

template<size_t D> static torch::nn::ConvTransposeOptions<D> convtran(K x,J i,Cast c) {
 torch::nn::ConvTransposeOptions<D> o(0,0,0);
 bool in=false,out=false,sz=false; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
   switch(j) {
    case 0: o.in_channels   (int64(x,i+j,c,Setting::in));        in=true; break;
    case 1: o.out_channels  (int64(x,i+j,c,Setting::in));       out=true; break;
    case 2: o.kernel_size   (exarray<D>(x,i+j,c,Setting::size)); sz=true; break;
    case 3: o.stride        (exarray<D>(x,i+j,c,Setting::stride)); break;
    case 4: o.padding       (exarray<D>(x,i+j,c,Setting::pad));    break;
    case 5: o.output_padding(exarray<D>(x,i+j,c,Setting::outpad)); break;
    case 6: o.groups        (int64(x,i+j,c,Setting::groups));      break;
    case 7: o.bias          (mbool(x,i+j,c,Setting::bias));        break;
    case 8: o.dilation      (exarray<D>(x,i+j,c,Setting::dilate)); break;
    case 9: o.padding_mode  (convpad(mode(x,i+j,c,Setting::padmode))); break;
    default: AT_ERROR(msym(c),": up to 9 positional arguments expected, ",n," given");
  }
 while(xpair(p))
  switch(mset(p.k)) {
   case Setting::in:        o.in_channels   (int64(p,c));      in=true; break;
   case Setting::out:       o.out_channels  (int64(p,c));     out=true; break;
   case Setting::size:      o.kernel_size   (exarray<D>(p,c)); sz=true; break;
   case Setting::stride:    o.stride        (exarray<D>(p,c)); break;
   case Setting::pad:       o.padding       (exarray<D>(p,c)); break;
   case Setting::outpad:    o.output_padding(exarray<D>(p,c)); break;
   case Setting::groups:    o.groups        (int64(p,c));      break;
   case Setting::bias:      o.bias          (mbool(p,c));      break;
   case Setting::dilate:    o.dilation      (exarray<D>(p,c)); break;
   case Setting::padmode:   o.padding_mode(convpad(mode(p,c)));break;
   default: AT_ERROR("Unrecognized convolution option: ",p.k); break;
  }
 TORCH_CHECK(in,  msym(c), ": number of input channels not defined");
 TORCH_CHECK(out, msym(c), ": number of output channels not defined");
 TORCH_CHECK(sz,  msym(c), ": no kernel size(s) given");
 return o;
}

template<size_t D> static void conv(bool a,K x,const torch::nn::detail::ConvNdOptions<D>& o) {
 torch::nn::detail::ConvNdOptions<D> d(o.in_channels(),o.out_channels(),o.kernel_size());
 bool t=o.transposed();
 OPTION(x, in,   kj(o.in_channels()));
 OPTION(x, out,  kj(o.out_channels()));
 OPTION(x, size, KEX(o.kernel_size()));
 if(a || (*o.stride()  != *d.stride()))  OPTION(x, stride, KEX(o.stride()));
 if(a || (*o.padding() != *d.padding())) OPTION(x, pad,    KEX(o.padding()));
 if(t) {
  if(a || (*o.output_padding() != *d.output_padding())) OPTION(x, outpad, KEX(o.output_padding()));
 } else {
  if(a || (*o.dilation() != *d.dilation())) OPTION(x, dilate, KEX(o.dilation()));
 }
 if(a || ( o.groups()    !=  d.groups())) OPTION(x, groups, kj(o.groups()));
 if(a || ( o.bias()      !=  d.bias()))   OPTION(x, bias,   kb(o.bias()));
 if(t) {
  if(a || (*o.dilation() != *d.dilation())) OPTION(x, dilate, KEX(o.dilation()));
 }
 if(a || o.padding_mode().index() != d.padding_mode().index()) OPTION(x, padmode, ks(ESYM(o.padding_mode())));
}

// --------------------------------------------------------------------------------------
// fold,unfold - set/get size,dilation,padding,stride
// --------------------------------------------------------------------------------------
static torch::nn::FoldOptions fold(K x,J i,Cast c) {
 torch::nn::FoldOptions o(0,0);
 bool out=false,sz=false; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
   switch(j) {
    case 0: o.output_size(exarray<2>(x,i+j,c,Setting::out)); out=true; break;
    case 1: o.kernel_size(exarray<2>(x,i+j,c,Setting::size)); sz=true; break;
    case 2: o.dilation   (exarray<2>(x,i+j,c,Setting::dilate)); break;
    case 3: o.padding    (exarray<2>(x,i+j,c,Setting::pad));    break;
    case 4: o.stride     (exarray<2>(x,i+j,c,Setting::stride)); break;
    default: AT_ERROR(msym(c),": up to 5 positional arguments expected, ",n," given");
  }
 while(xpair(p))
  switch(mset(p.k)) {
   case Setting::out:       o.output_size(exarray<2>(p,c));out=true; break;
   case Setting::size:      o.kernel_size(exarray<2>(p,c)); sz=true; break;
   case Setting::dilate:    o.dilation   (exarray<2>(p,c)); break;
   case Setting::pad:       o.padding    (exarray<2>(p,c)); break;
   case Setting::stride:    o.stride     (exarray<2>(p,c)); break;
   default: AT_ERROR("Unrecognized fold option: ",p.k); break;
  }
 TORCH_CHECK(out, msym(c),": no output size given");
 TORCH_CHECK(sz,  msym(c),": no kernel size given");
 return o;
}

static void fold(bool a,K x,const torch::nn::FoldOptions& o) {
 torch::nn::FoldOptions d(o.output_size(),o.kernel_size());
 OPTION(x, out,  KEX(o.output_size()));
 OPTION(x, size, KEX(o.kernel_size()));
 if(a || (*o.dilation() != *d.dilation())) OPTION(x, dilate, KEX(o.dilation()));
 if(a || (*o.padding()  != *d.padding()))  OPTION(x, pad,    KEX(o.padding()));
 if(a || (*o.stride()   != *d.stride()))   OPTION(x, stride, KEX(o.stride()));
}

static torch::nn::UnfoldOptions unfold(K x,J i,Cast c) {
 torch::nn::UnfoldOptions o(0);
 bool sz=false; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
   switch(j) {
    case 0: o.kernel_size(exarray<2>(x,i+j,c,Setting::size)); sz=true; break;
    case 1: o.dilation   (exarray<2>(x,i+j,c,Setting::dilate)); break;
    case 2: o.padding    (exarray<2>(x,i+j,c,Setting::pad));    break;
    case 3: o.stride     (exarray<2>(x,i+j,c,Setting::stride)); break;
    default: AT_ERROR(msym(c),": up to 4 positional arguments expected, ",n," given");
  }
 while(xpair(p))
  switch(mset(p.k)) {
   case Setting::size:      o.kernel_size(exarray<2>(p,c)); sz=true; break;
   case Setting::dilate:    o.dilation   (exarray<2>(p,c)); break;
   case Setting::pad:       o.padding    (exarray<2>(p,c)); break;
   case Setting::stride:    o.stride     (exarray<2>(p,c)); break;
   default: AT_ERROR("Unrecognized unfold option: ",p.k); break;
  }
 TORCH_CHECK(sz, msym(c),": no kernel size given");
 return o;
}

static void unfold(bool a,K x,const torch::nn::UnfoldOptions& o) {
 torch::nn::UnfoldOptions d(o.kernel_size());
 OPTION(x, size, KEX(o.kernel_size()));
 if(a || (*o.dilation() != *d.dilation())) OPTION(x, dilate, KEX(o.dilation()));
 if(a || (*o.padding()  != *d.padding()))  OPTION(x, pad,    KEX(o.padding()));
 if(a || (*o.stride()   != *d.stride()))   OPTION(x, stride, KEX(o.stride()));
}

static K kfold(K x,Cast c) {
 KTRY
  TORCH_CHECK(!x->t, msym(c)," not implemented for ",kname(x->t));
  Tensor r, *t=xten(x,0);
  return kresult(t, c==Cast::fold
       ? torch::nn::functional::fold  (t ? *t : kput(x,0),   fold(x,1,c))
       : torch::nn::functional::unfold(t ? *t : kput(x,0), unfold(x,1,c)));
 KCATCH("fold");
}

KAPI   Fold(K x) {return kfold(x, Cast::fold);}
KAPI Unfold(K x) {return kfold(x, Cast::unfold);}

// --------------------------------------------------------------------------------------
// drop - create dropout module given probability/set dictionary given module
// --------------------------------------------------------------------------------------
static torch::nn::DropoutOptions drop(K x,J i,Cast c) {
 torch::nn::DropoutOptions o; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
   switch(j) {
    case 0: o.p(mdouble(x,i+j,c,Setting::p)); break;
    case 1: o.inplace(mbool(x,i+j,c,Setting::inplace)); break;
    default: AT_ERROR(msym(c),": up to 2 positional arguments expected, ",n," given");
  }
 while(xpair(p))
  switch(mset(p.k)) {
   case Setting::p: o.p(mdouble(p,c)); break;
   case Setting::inplace: o.inplace(mbool(p,c)); break;
   default: AT_ERROR("Unrecognized dropout option: ",p.k); break;
  }
 return o;
}

static void drop(bool a,K x,const torch::nn::DropoutOptions& o) {
 torch::nn::DropoutOptions d;
 if(a || o.p()       != d.p())       OPTION(x, p,       kf(o.p()));
 if(a || o.inplace() != d.inplace()) OPTION(x, inplace, kb(o.inplace()));
}

// --------------------------------------------------------------------------------------
// create embedding/embedding bag module given options:
// embedmode - translate symbol to internal embedding mode (variant)
// embedset - set name/value pairs specific to Embedding vs EmbeddingBag
// embedpair - handle name/value pairs for both types of embedding modules
// embedwt - handle options depending on whether pre-trained weights supplied
// embed, embedbag - process args and return Embedding/EmbeddingBag module
// --------------------------------------------------------------------------------------
static torch::nn::EmbeddingBagMode embedmode(S s) {
 switch(emap(s)) {
  case Enum::sum:  return torch::kSum;
  case Enum::mean: return torch::kMean;
  case Enum::max:  return torch::kMax;
  default: AT_ERROR("unrecognized mode for embedding bag: ",s);
 }
}

static void embedset(Cast c,Setting s,Pairs& p,torch::nn::EmbeddingOptions& o) {
 if(s == Setting::padindex) o.padding_idx(int64n(p,c));
 else AT_ERROR("Unrecognized option for ",msym(c),": ",mset(s));
}

static void embedset(Cast c,Setting s,Pairs& p,torch::nn::EmbeddingBagOptions& o) {
 if       (s == Setting::mode) o.mode(embedmode(psym(p)));
 else if(s == Setting::lastoffset) o.include_last_offset(mbool(p,c));
 else AT_ERROR("Unrecognized option for ",msym(c),": ",mset(s));
}

template<typename O> static void embedpair(Cast c,Pairs& p,O& o,Tensor& w,bool &z) {
 while(xpair(p))
  switch(mset(p.k)) {
   case Setting::rows:       o.num_embeddings(int64(p,c)); break;
   case Setting::cols:       o.embedding_dim (int64(p,c)); break;
   case Setting::padindex:   embedset(c,Setting::padindex,p,o); break;
   case Setting::maxnorm:    o.max_norm(optdouble(p,c)); break;
   case Setting::p:          o.norm_type(mdouble(p,c)); break;
   case Setting::scale:      o.scale_grad_by_freq(mbool(p,c)); break;
   case Setting::sparse:     o.sparse(mbool(p,c)); break;
   case Setting::weight:     if(!pempty(p)) pten(p,w); break;
   case Setting::freeze:     z=mbool(p,c); break;
   case Setting::mode:       embedset(c,Setting::mode,p,o); break;
   case Setting::lastoffset: embedset(c,Setting::lastoffset,p,o);
   default: AT_ERROR("Embedding option: ",p.k," unrecognized");
  }
}

template<typename M,typename O> static M embedwt(Cast c,O o,const Tensor& w,bool z) {
 bool a=o.num_embeddings()!=nj,b=o.embedding_dim()!=nj;
 if(w.defined()) {
  TORCH_CHECK(w.dim()==2, msym(c),": ",w.dim(),"-dim weights given, 2-dim matrix expected");
  TORCH_CHECK(w.is_floating_point(), msym(c),": weight matrix is not floating point");
  if(!a) o.num_embeddings(w.size(0));
  else TORCH_CHECK(o.num_embeddings()==w.size(0), "rows = ",o.num_embeddings()," but weights are ",w.sizes());
  if(!b) o.embedding_dim(w.size(1));
  else TORCH_CHECK(o.embedding_dim() ==w.size(1), "cols = ",o.embedding_dim(), " but weights are ",w.sizes());
   M m=M(o._weight(w));
   m->weight.set_requires_grad(!z);
   return m;
 } else {
  TORCH_CHECK(a,"embed: supply number of rows in the embedding matrix");
  TORCH_CHECK(b,"embed: supply number of cols in the embedding matrix");
  return M(o);
 }
}

static torch::nn::Embedding embed(K x,J i,Cast c) {
 bool z=false; Pairs p; Tensor w; J n=xargc(x,i,p);
 torch::nn::EmbeddingOptions o(nj,nj);
 for(J j=0;j<n;++j) {
   switch(j) {
    case 0:
     switch(kK(x)[i+j]->t) {
      case 0:   if(!xten(x,i+j,w)) w=kput(x,i+j); break;
      case -KJ: o.num_embeddings(int64(x,i+j,c,Setting::rows)); break;
      default:  AT_ERROR("embed: 1st arg is number of rows or weight matrix");
     }
     break;
    case 1: 
     if(w.defined()) z=mbool(x,i+j,c,Setting::freeze);
     else  o.embedding_dim(int64(x,i+j,c,Setting::cols));
     break;
    case 2: o.padding_idx(int64n(x,i+j,c,Setting::padindex)); break;
    case 3: o.max_norm(optdouble(x,i+j,c,Setting::maxnorm)); break;
    case 4: o.norm_type(mdouble(x,i+j,c,Setting::p)); break;
    case 5: o.scale_grad_by_freq(mbool(x,i+j,c,Setting::scale)); break;
    case 6: o.sparse(mbool(x,i+j,c,Setting::sparse)); break;
    default: AT_ERROR(msym(c),": up to 7 positional arguments expected, ",n," given");
  }
 }
 embedpair(c,p,o,w,z);
 return embedwt<torch::nn::Embedding,torch::nn::EmbeddingOptions>(c,o,w,z);
}

static torch::nn::EmbeddingBag embedbag(K x,J i,Cast c) {
 bool z=false; Pairs p; Tensor w; J n=xargc(x,i,p);
 torch::nn::EmbeddingBagOptions o(nj,nj);
 // allow mode if last arg even if early in sequence
 if(!x->t && n>1 && n<6 && xsym(x,i+n-1))
  n--, o.mode(embedmode(mode(x,i+n,c,Setting::mode)));
 for(J j=0;j<n;++j) {
   switch(j) {
    case 0:
     switch(kK(x)[i+j]->t) {
      case 0:   if(!xten(x,i+j,w)) w=kput(x,i+j); break;
      case -KJ: o.num_embeddings(int64(x,i+j,c,Setting::rows)); break;
      default:  AT_ERROR("embed: 1st arg is number of rows or weight matrix");
     }
     break;
    case 1: 
     if(w.defined()) z=mbool(x,i+j,c,Setting::freeze);
     else  o.embedding_dim(int64(x,i+j,c,Setting::cols));
     break;
    case 2: o.max_norm(optdouble(x,i+j,c,Setting::maxnorm)); break;
    case 3: o.norm_type(mdouble(x,i+j,c,Setting::p)); break;
    case 4: o.scale_grad_by_freq(mbool(x,i+j,c,Setting::scale)); break;
    case 5: o.mode(embedmode(mode(x,i+j,c,Setting::mode))); break;
    case 6: o.sparse(mbool(x,i+j,c,Setting::sparse)); break;
    default: AT_ERROR(msym(c),": up to 7 positional arguments expected, ",n," given");
  }
 }
 embedpair(c,p,o,w,z);
 return embedwt<torch::nn::EmbeddingBag,torch::nn::EmbeddingBagOptions>(c,o,w,z);
}

// -----------------------------------------------------------------------------------------
// retrieve settings from existing Embedding/EmbeddingBag:
// embedget - templated fucntion to retrieve options specific to Embedding or EmbeddingBag
// embed - templated function which gets options and initial optional weights
// -----------------------------------------------------------------------------------------
static void embedget(bool a,K x,Cast c,Setting s,const torch::nn::EmbeddingOptions& o,const torch::nn::EmbeddingOptions& d) {
 if(s == Setting::padindex && (a || o.padding_idx().has_value()))
  OPTION(x, padindex, kj(o.padding_idx() ? o.padding_idx().value() : nj));
}

static void embedget(bool a,K x,Cast c,Setting s,const torch::nn::EmbeddingBagOptions& o,const torch::nn::EmbeddingBagOptions& d) {
 if(s == Setting::mode && (a || o.mode().index() != d.mode().index()))
  OPTION(x, mode, ks(ESYM(o.mode())));
 /*
 else if(s == Setting::lastoffset && (a || o.include_last_offset() != d.include_last_offset())
  OPTION(x, lastoffset, kb(o.include_last_offset()));
 */
}

template<typename O>static void embed(bool a,K x,Cast c,const O& o,const Tensor& w) {
 O d(o.num_embeddings(),o.embedding_dim());
 if(o._weight().defined()) {
  OPTION(x, weight, kget(o._weight()));
  OPTION(x, freeze, kb(!w.requires_grad()));
 } else {
  OPTION(x, rows, kj(o.num_embeddings()));
  OPTION(x, cols, kj(o.embedding_dim()));
 }
 embedget(a,x,c,Setting::padindex,o,d); // embedding only
 if(a || o.max_norm().has_value())                         OPTION(x, maxnorm, kf(o.max_norm() ? o.max_norm().value() : nf));
 if(a || o.norm_type()          != d.norm_type())          OPTION(x, p,       kf(o.norm_type()));
 if(a || o.scale_grad_by_freq() != d.scale_grad_by_freq()) OPTION(x, scale,   kb(o.scale_grad_by_freq()));
 embedget(a,x,c,Setting::mode,o,d); //EmbeddingBag only
 if(a || o.sparse()             != d.sparse())             OPTION(x, sparse,  kb(o.sparse()));
 //embedget(a,x,c,Setting::lastoffset,o,d);
}

// --------------------------------------------------------------------------------------
// linear - parse/retrieve args, invoke functional form
// --------------------------------------------------------------------------------------
static torch::nn::LinearOptions linear(K x,J i,Cast c) {
 bool b=true; int64_t in=nj,out=nj; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j) {
   switch(j) {
    case 0:  in=int64(x,i+j,c,Setting::in);   break;
    case 1: out=int64(x,i+j,c,Setting::out);  break;
    case 2:   b=mbool(x,i+j,c,Setting::bias); break;
    default: AT_ERROR(msym(c),": up to 3 positional arguments expected, ",n," given");
  }
 }
 while(xpair(p))
  switch(mset(p.k)) {
   case Setting::in:   in=int64(p,c); break;
   case Setting::out:  out=int64(p,c); break;
   case Setting::bias: b=mbool(p,c); break;
   default: AT_ERROR("Unrecognized linear option: ",p.k); break;
  }
 TORCH_CHECK(in>0,  msym(c), ": positive input size required");
 TORCH_CHECK(out>0, msym(c), ": positive output size required");
 return torch::nn::LinearOptions(in,out).bias(b);
}

static void linear(bool a,K x,const torch::nn::LinearImpl *m) {
 torch::nn::LinearOptions o=m->options, d(o.in_features(),o.out_features());
 OPTION(x, in,  kj(o.in_features()));
 OPTION(x, out, kj(o.out_features()));
 if(a || (o.bias() != d.bias())) OPTION(x, bias, kb(o.bias()));
}

KAPI Linear(K x) {
 KTRY
  TORCH_CHECK(!x->t, "linear not implemented for ",kname(x->t));
  TORCH_CHECK(x->n==2 || x->n==3, "linear requires 2-3 args, (input; weight; optional bias)");
  Tensor r, *a=xten(x,0), *w=xten(x,1), *b=xten(x,2);
  if(x->n==2)
   r=torch::linear(a ? *a : kput(x,0), w ? *w : kput(x,1));
  else
   r=torch::linear(a ? *a : kput(x,0), w ? *w : kput(x,1), b ? *b : kput(x,2));
  return kresult(a||w||b, r);
 KCATCH("linear");
}

// --------------------------------------------------------------------------------------
// bilinear - parse/retrieve args, callable functional form
// --------------------------------------------------------------------------------------
static torch::nn::BilinearOptions bilinear(K x,J i,Cast c) {
 bool b=true; int64_t in1=nj,in2=nj,out=nj; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j) {
   switch(j) {
    case 0: in1=int64(x,i+j,c,Setting::in1);   break;
    case 1: in2=int64(x,i+j,c,Setting::in2);   break;
    case 2: out=int64(x,i+j,c,Setting::out);  break;
    case 3:   b=mbool(x,i+j,c,Setting::bias); break;
    default: AT_ERROR(msym(c),": up to 4 positional arguments(in1,in2,out,biasflag) expected, ",n," given");
  }
 }
 while(xpair(p))
  switch(mset(p.k)) {
   case Setting::in1:  in1=int64(p,c); break;
   case Setting::in2:  in2=int64(p,c); break;
   case Setting::out:  out=int64(p,c); break;
   case Setting::bias: b=mbool(p,c); break;
   default: AT_ERROR("Unrecognized bilinear option: ",p.k); break;
  }
 TORCH_CHECK(in1>0 && in2>0, msym(c), ": positive input sizes required");
 TORCH_CHECK(out>0,          msym(c), ": positive output size required");
 return torch::nn::BilinearOptions(in1,in2,out).bias(b);
}

static void bilinear(bool a,K x,const torch::nn::BilinearImpl *m) {
 torch::nn::BilinearOptions o=m->options, d(o.in1_features(),o.in2_features(),o.out_features());
 OPTION(x, in1,  kj(o.in1_features()));
 OPTION(x, in2,  kj(o.in2_features()));
 OPTION(x, out, kj(o.out_features()));
 if(a || (o.bias() != d.bias())) OPTION(x, bias, kb(o.bias()));
}

KAPI Bilinear(K x) {
 KTRY
  TORCH_CHECK(!x->t, "bilinear not implemented for ",kname(x->t));
  TORCH_CHECK(x->n==3 || x->n==4, "blinear requires 3-4 args, (input1; input2; weight; optional bias)");
  Tensor r, *x1=xten(x,0), *x2=xten(x,1), *w=xten(x,2), *b=xten(x,3);
  return kresult(x1||x2||w||b, torch::bilinear(x1 ? *x1 : kput(x,0),
                                               x2 ? *x2 : kput(x,1),
                                                w ?  *w : kput(x,2),
                                                x->n==3 ? Tensor{} : (b ? *b : kput(x,3))));
 KCATCH("bilinear");
}


// --------------------------------------------------------------------------------------
// rnn - create rnn/gru/lstm module given options/set dictionary of options from module
// --------------------------------------------------------------------------------------
template<typename M,typename O>
static M rnn(Cast c,K x,J k) {
 // PATCH: auto f=torch::nn::RNNActivation::ReLU;
 bool b=true,bi=false,ba=false; Pairs p; J i=-1,h=-1,l=1,n=xargc(x,k,p); double d=0.0;
 if(!((n==0 && p.n) || (xlong(x,k,i) && (n==1 || (n==2 && xlong(x,k+1,h))))))
  AT_ERROR("Unrecognized arguments for ",msym(c)," module");
 // PATCH: bool r=std::is_same<M,torch::nn::RNN>::value;
 while(xpair(p))
  switch(mset(p.k)) {
   case Setting::in:          i=plong(p); break;
   case Setting::hidden:      h=plong(p); break;
   case Setting::layers:      l=plong(p); break;
   case Setting::bias:        b=pbool(p); break;
   case Setting::bi:         bi=pbool(p); break;
   case Setting::batchfirst: ba=pbool(p); break;
   case Setting::dropout:   d=pdouble(p); break;
   // PATCH: case Setting::fn: if(r) f=rnnfn(psym(p)); else AT_ERROR("activation function only for RNN module"); break;
   default: AT_ERROR(msym(c)," option: ",p.k," unrecognized, expected one of in,hidden,layers,bias,bi,batchfirst,drop,fn");
  }
 // PATCH: layers -> num_layers, with_bias -> bias
 auto o=O(i,h).num_layers(l).dropout(d).bias(b).bidirectional(bi).batch_first(ba);
 // PATCH: if(r) rnnfn(o,f);
 return M(o);
}

template<typename M,typename O>
static void rnn(bool a,K x,const M* m) {
 O o=m->options, d(o.input_size(),o.hidden_size()); // PATCH: S f=rnnfn(o);
 OPTION(x, in,     kj(o.input_size()));
 OPTION(x, hidden, kj(o.hidden_size()));
 if(a || (o.num_layers()    != d.num_layers()))   OPTION(x, layers,     kj(o.num_layers()));
 if(a || (o.dropout()       != d.dropout()))      OPTION(x, dropout,    kf(o.dropout()));
 // PATCH: if((a && f) || f           != rnnfn(d))          OPTION(x, fn,         ks(f));
 if(a || (o.bias()          != d.bias()))         OPTION(x, bias,       kb(o.bias()));
 if(a || (o.bidirectional() != d.bidirectional()))OPTION(x, bi,         kb(o.bidirectional()));
 if(a || (o.batch_first()   != d.batch_first()))  OPTION(x, batchfirst, kb(o.batch_first()));
}

// ----------------------------------------------------------------------------------
//  maxpool - process args, return dictionary of options, call functional form
// ----------------------------------------------------------------------------------
template<size_t D> static torch::nn::MaxPoolOptions<D> maxpool(K x,J i,Cast c) {
 torch::nn::MaxPoolOptions<D> o(0);
 bool sz=false,st=false; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j) {
   switch(j) {
    case 0: o.kernel_size(exarray<D>(x,i+j,c,Setting::size));    sz=true; break;
    case 1: o.stride     (exarray<D>(x,i+j,c,Setting::stride));  st=true; break;
    case 2: o.padding    (exarray<D>(x,i+j,c,Setting::pad));     break;
    case 3: o.dilation   (exarray<D>(x,i+j,c,Setting::dilate));  break;
    case 4: o.ceil_mode  (mbool     (x,i+j,c,Setting::ceiling)); break;
    default: AT_ERROR(msym(c),": up to 5 positional arguments expected, ",n," given");
  }
 }
 while(xpair(p))
  switch(mset(p.k)) {
   case Setting::size:    o.kernel_size(exarray<D>(p,c)); sz=true; break;
   case Setting::stride:  o.stride     (exarray<D>(p,c)); st=true; break;
   case Setting::pad:     o.padding    (exarray<D>(p,c)); break;
   case Setting::dilate:  o.dilation   (exarray<D>(p,c)); break;
   case Setting::ceiling: o.ceil_mode  (mbool(p,c)); break;
   default: AT_ERROR("Unrecognized max pooling option: ",p.k); break;
  }
 TORCH_CHECK(sz, msym(c),": no kernel size given");
 if(!st) o.stride(o.kernel_size());
 return o;
}

template<size_t D,typename M> static void maxpool(bool a,K x,const M* m) {
 torch::nn::MaxPoolOptions<D> o=m->options, d(o.kernel_size());
 OPTION(x, size, KEX(o.kernel_size()));
 if(a || *o.stride()   != *d.stride())   OPTION(x, stride,  KEX(o.stride()));
 if(a || *o.padding()  != *d.padding())  OPTION(x, pad,     KEX(o.padding()));
 if(a || *o.dilation() != *d.dilation()) OPTION(x, dilate,  KEX(o.dilation()));
 if(a || o.ceil_mode() != d.ceil_mode()) OPTION(x, ceiling, kb(o.ceil_mode()));
}

// ----------------------------------------------------------------------------------
//  avgpool - process args, return dictionary of options, call functional form
// ----------------------------------------------------------------------------------
template<size_t D> static torch::nn::AvgPoolOptions<D> avgpool(K x,J i,Cast c) {
 torch::nn::AvgPoolOptions<D> o(0);
 bool sz=false,st=false; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j) {
   switch(j) {
    case 0: o.kernel_size      (exarray<D>(x,i+j,c,Setting::size));   sz=true; break;
    case 1: o.stride           (exarray<D>(x,i+j,c,Setting::stride)); st=true; break;
    case 2: o.padding          (exarray<D>(x,i+j,c,Setting::pad));      break;
    case 3: o.ceil_mode        (mbool     (x,i+j,c,Setting::ceiling));  break;
    case 4: o.count_include_pad(mbool     (x,i+j,c,Setting::countpad)); break;
    case 5: o.divisor_override (int64n    (x,i+j,c,Setting::divisor));  break;
    default: AT_ERROR(msym(c),": up to 6 positional arguments expected, ",n," given");
  }
 }
 while(xpair(p))
  switch(mset(p.k)) {
   case Setting::size:    o.kernel_size (exarray<D>(p,c)); sz=true; break;
   case Setting::stride:  o.stride      (exarray<D>(p,c)); st=true; break;
   case Setting::pad:     o.padding     (exarray<D>(p,c)); break;
   case Setting::ceiling: o.ceil_mode        (mbool(p,c)); break;
   case Setting::countpad:o.count_include_pad(mbool(p,c)); break;
   case Setting::divisor: o.divisor_override(int64n(p,c)); break;
   default: AT_ERROR("Unrecognized avg pooling option: ",p.k); break;
  }
 TORCH_CHECK(sz, msym(c),": no kernel size given");
 if(!st) o.stride(o.kernel_size());
 return o;
}

template<size_t D,typename M> static void avgpool(bool a,K x,const M* m) {
 torch::nn::AvgPoolOptions<D> o=m->options, d(o.kernel_size());
 OPTION(x, size, KEX(o.kernel_size()));
 if(a || *o.stride()           != *d.stride())           OPTION(x, stride,   KEX(o.stride()));
 if(a || *o.padding()          != *d.padding())          OPTION(x, pad,      KEX(o.padding()));
 if(a || o.ceil_mode()         != d.ceil_mode())         OPTION(x, ceiling,  kb(o.ceil_mode()));
 if(a || o.count_include_pad() != d.count_include_pad()) OPTION(x, countpad, kb(o.count_include_pad()));
 if(a || o.divisor_override().has_value())               OPTION(x, divisor,  kj(o.divisor_override() ? o.divisor_override().value() : nj));
}

// ---------------------------------------------------------------------------------------
// adaptive pooling - process args, return dictionary of options, call functional form
// adapt - multiple versions to handle expanding array(1d) vs array of optionals(2,3d)
// ---------------------------------------------------------------------------------------
template<size_t D> static void adapt(ExpandingArray<D>& a,K x,J i,Cast c)        {a=exarray<D>(x,i,c,Setting::size);}
template<size_t D> static void adapt(ExpandingArray<D>& a,const Pairs& p,Cast c) {a=exarray<D>(p,c);}
template<size_t D> static void adapt(Exoptional<D>& a,K x,J i,Cast c)        {a=exoptional<D>(x,i,c,Setting::size);}
template<size_t D> static void adapt(Exoptional<D>& a,const Pairs& p,Cast c) {a=exoptional<D>(p,c);}

template<size_t D> static bool adapt(ExpandingArray<D>& a) {for(auto &v:*a) if(v != nj) return true; return false;}
template<size_t D> static bool adapt(Exoptional<D>& a)     {for(auto &v:*a) if(v)       return true; return false;}

template<size_t D,typename T> static T adapt(K x,J i,Cast c) {
 T o(0); bool sz=false; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j) {
   switch(j) {
    case 0: adapt<D>(o.output_size(),x,i+j,c); sz=true; break;
    default: AT_ERROR(msym(c),": 1 positional argument expected, ",n," given");
  }
 }
 while(xpair(p))
  switch(mset(p.k)) {
   case Setting::size: adapt<D>(o.output_size(),p,c); sz=true; break;
   default: AT_ERROR("Unrecognized ",msym(c)," option: ",p.k); break;
  }
 TORCH_CHECK(sz, msym(c),": no output size given");
 TORCH_CHECK(adapt(o.output_size()), msym(c),": no output size");
 return o;
}

template<typename M> static void adapt(K x,const M* m) {
 OPTION(x, size, KEX(m->options.output_size()));
}

// ----------------------------------------------------------------------------------
// fpool - fractional max pooling for 2 & 3d layers
// ----------------------------------------------------------------------------------
template<size_t D> static torch::nn::FractionalMaxPoolOptions<D> fpool(K x,J i,Cast c) {
 torch::nn::FractionalMaxPoolOptions<D> o(0);
 bool e,sz=false; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j) {
   e=xempty(x,i+j);
   switch(j) {
    case 0: o.kernel_size(exarray<D>(x,i+j,c,Setting::size)); sz=true; break;
    case 1: if(e) o.output_size( c10::nullopt); else o.output_size ( exarray<D>(x,i+j,c,Setting::outsize)); break;
    case 2: if(e) o.output_ratio(c10::nullopt); else o.output_ratio(exdouble<D>(x,i+j,c,Setting::ratio));   break;
    default: AT_ERROR(msym(c),": up to 3 positional arguments expected, ",n," given");
  }
 }
 while(xpair(p)) {
  e=pempty(p);
  switch(mset(p.k)) {
   case Setting::size:    o.kernel_size(exarray<D>(p,c)); sz=true; break;
   case Setting::outsize: if(e) o.output_size (c10::nullopt); else o.output_size(exarray  <D>(p,c)); break;
   case Setting::ratio:   if(e) o.output_ratio(c10::nullopt); else o.output_ratio(exdouble<D>(p,c)); break;
   default: AT_ERROR("Unrecognized ",msym(c)," option: ",p.k); break;
  }
 }
 TORCH_CHECK(sz, msym(c), ": no kernel size given");
 TORCH_CHECK(o.output_size()||o.output_ratio(), msym(c), ": no output size or ratio given");
 TORCH_CHECK(!(o.output_size()&&o.output_ratio()), msym(c), ": cannot specify both output size & output ratio");
 return o;
}

template<size_t D,typename M> static void fpool(bool a,K x,const M* m) {
 torch::nn::FractionalMaxPoolOptions<D> o=m->options;
 OPTION(x, size, KEX(o.kernel_size()));
 if(a || o.output_size().has_value())    OPTION(x, outsize, o.output_size() ? KEX(o.output_size().value())  : ktn(0,0));
 if(a || o.output_ratio().has_value())   OPTION(x, ratio,   o.output_ratio()? KEX(o.output_ratio().value()) : ktn(0,0));
}

// ----------------------------------------------------------------------------------
// lppool - power-average pooling
// ----------------------------------------------------------------------------------
template<size_t D> static torch::nn::LPPoolOptions<D> lppool(K x,J i,Cast c) {
 torch::nn::LPPoolOptions<D> o(0,0);
 bool pw=false,sz=false,st=false; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j) {
   switch(j) {
    case 0: o.norm_type  (mdouble(x,i+j,c,Setting::p));         pw=true; break;
    case 1: o.kernel_size(exarray<D>(x,i+j,c,Setting::size));   sz=true; break;
    case 2: o.stride     (exarray<D>(x,i+j,c,Setting::stride)); st=true; break;
    case 3: o.ceil_mode  (mbool    (x,i+j,c,Setting::ceiling)); break;
    default: AT_ERROR(msym(c),": up to 4 positional arguments expected, ",n," given");
  }
 }
 while(xpair(p))
  switch(mset(p.k)) {
   case Setting::p:       o.norm_type  (mdouble   (p,c)); pw=true; break;
   case Setting::size:    o.kernel_size(exarray<D>(p,c)); sz=true; break;
   case Setting::stride:  o.stride     (exarray<D>(p,c)); st=true; break;
   case Setting::ceiling: o.ceil_mode  (mbool(p,c)); break;
   default: AT_ERROR("Unrecognized ",msym(c)," option: ",p.k); break;
  }
 TORCH_CHECK(pw, msym(c),": no power given");
 TORCH_CHECK(sz, msym(c),": no kernel size given");
 if(!st) o.stride(o.kernel_size());
 return o;
}

template<size_t D,typename M> static void lppool(bool a,K x,const M* m) {
 torch::nn::LPPoolOptions<D> o=m->options, d(o.norm_type(),o.kernel_size());
 OPTION(x, p,    kf(o.norm_type()));
 OPTION(x, size, KEX(o.kernel_size()));
 if(a || *o.stride()   != *d.stride())   OPTION(x, stride,  KEX(o.stride()));
 if(a || o.ceil_mode() != d.ceil_mode()) OPTION(x, ceiling, kb(o.ceil_mode()));
}

// ----------------------------------------------------------------------------------
// functional form of pooling methods:
// ----------------------------------------------------------------------------------
static K pool(K x,Cast c) {
 KTRY
  TORCH_CHECK(!x->t, msym(c)," not implemented for ",kname(x->t));
  Tensor r, *t=xten(x,0);
  switch(c) {
   case Cast::maxpool1d:  r=torch::nn::functional::max_pool1d(t ? *t : kput(x,0), maxpool<1>(x,1,c)); break;
   case Cast::maxpool2d:  r=torch::nn::functional::max_pool2d(t ? *t : kput(x,0), maxpool<2>(x,1,c)); break;
   case Cast::maxpool3d:  r=torch::nn::functional::max_pool3d(t ? *t : kput(x,0), maxpool<3>(x,1,c)); break;
   case Cast::avgpool1d:  r=torch::nn::functional::avg_pool1d(t ? *t : kput(x,0), avgpool<1>(x,1,c)); break;
   case Cast::avgpool2d:  r=torch::nn::functional::avg_pool2d(t ? *t : kput(x,0), avgpool<2>(x,1,c)); break;
   case Cast::avgpool3d:  r=torch::nn::functional::avg_pool3d(t ? *t : kput(x,0), avgpool<3>(x,1,c)); break;
   case Cast::adaptmax1d: r=torch::nn::functional::adaptive_max_pool1d(t ? *t : kput(x,0), adapt<1,torch::nn::AdaptiveMaxPool1dOptions>(x,1,c)); break;
   case Cast::adaptmax2d: r=torch::nn::functional::adaptive_max_pool2d(t ? *t : kput(x,0), adapt<2,torch::nn::AdaptiveMaxPool2dOptions>(x,1,c)); break;
   case Cast::adaptmax3d: r=torch::nn::functional::adaptive_max_pool3d(t ? *t : kput(x,0), adapt<3,torch::nn::AdaptiveMaxPool3dOptions>(x,1,c)); break;
   case Cast::adaptavg1d: r=torch::nn::functional::adaptive_avg_pool1d(t ? *t : kput(x,0), adapt<1,torch::nn::AdaptiveAvgPool1dOptions>(x,1,c)); break;
   case Cast::adaptavg2d: r=torch::nn::functional::adaptive_avg_pool2d(t ? *t : kput(x,0), adapt<2,torch::nn::AdaptiveAvgPool2dOptions>(x,1,c)); break;
   case Cast::adaptavg3d: r=torch::nn::functional::adaptive_avg_pool3d(t ? *t : kput(x,0), adapt<3,torch::nn::AdaptiveAvgPool3dOptions>(x,1,c)); break;
   case Cast::fmaxpool2d: r=torch::nn::functional::fractional_max_pool2d(t ? *t : kput(x,0), fpool<2>(x,1,c)); break;
   case Cast::fmaxpool3d: r=torch::nn::functional::fractional_max_pool3d(t ? *t : kput(x,0), fpool<3>(x,1,c)); break;
   case Cast::lppool1d:   r=torch::nn::functional::lp_pool1d(t ? *t : kput(x,0), lppool<1>(x,1,c)); break;
   case Cast::lppool2d:   r=torch::nn::functional::lp_pool2d(t ? *t : kput(x,0), lppool<2>(x,1,c)); break;
   default: AT_ERROR("Unrecognized pooling function");
  }
  return kresult(t,r);
 KCATCH("pool");
}

KAPI maxpool1d(K x)  {return pool(x,Cast::maxpool1d);}
KAPI maxpool2d(K x)  {return pool(x,Cast::maxpool2d);}
KAPI maxpool3d(K x)  {return pool(x,Cast::maxpool3d);}
KAPI avgpool1d(K x)  {return pool(x,Cast::avgpool1d);}
KAPI avgpool2d(K x)  {return pool(x,Cast::avgpool2d);}
KAPI avgpool3d(K x)  {return pool(x,Cast::avgpool3d);}
KAPI adaptmax1d(K x) {return pool(x,Cast::adaptmax1d);}
KAPI adaptmax2d(K x) {return pool(x,Cast::adaptmax2d);}
KAPI adaptmax3d(K x) {return pool(x,Cast::adaptmax3d);}
KAPI adaptavg1d(K x) {return pool(x,Cast::adaptavg1d);}
KAPI adaptavg2d(K x) {return pool(x,Cast::adaptavg2d);}
KAPI adaptavg3d(K x) {return pool(x,Cast::adaptavg3d);}
KAPI fmaxpool2d(K x) {return pool(x,Cast::fmaxpool2d);}
KAPI fmaxpool3d(K x) {return pool(x,Cast::fmaxpool3d);}
KAPI lppool1d(K x)   {return pool(x,Cast::lppool1d);}
KAPI lppool2d(K x)   {return pool(x,Cast::lppool2d);}

// ----------------------------------------------------------------------------------
// padmode - match k symbol to std::variant style enumeration
// pad - n-dimensional padding, specify even number of sizes and optional pad value
// ----------------------------------------------------------------------------------
static void padmode(torch::nn::functional::PadFuncOptions& o,S s) {
 switch(emap(s)) {
  case Enum::constant:  o.mode(torch::kConstant); break;
  case Enum::reflect:   o.mode(torch::kReflect); break;
  case Enum::replicate: o.mode(torch::kReplicate); break;
  case Enum::circular:  o.mode(torch::kCircular); break;
  default: AT_ERROR("unrecognized padding mode: ",s); break;
 }
}

static torch::nn::functional::PadFuncOptions pad(K x,J i,Cast c) {
 torch::nn::functional::PadFuncOptions o({}); S s; Pairs p; J n=xargc(x,i,p); IntArrayRef a;
 for(J j=0;j<n;++j)
   switch(j) {
    case 0: TORCH_CHECK(xsize(x,i+j,a), msym(c),": expecting 1st arg of padding size(s)"); break;
    case 1:
     if(xsym(x,i+j,s)) padmode(o,s);
     else if(n==2)     o.value(mdouble(x,i+j,c,Setting::value));
     else AT_ERROR("pad: unrecognized 2nd arg, expecting mode or value");
     break;
    case 2: o.value(mdouble(x,i+j,c,Setting::value)); break;
    default: AT_ERROR(msym(c),": up to 3 positional args expected(padding;mode;value), ",n," given");
  }
 while(xpair(p))
  switch(mset(p.k)) {
   case Setting::pad:    psize(p,a); break;
   case Setting::mode:   padmode(o,psym(p)); break;
   case Setting::value:  o.value(mdouble(p,c)); break;
   default: AT_ERROR("padding option: ",p.k," not recognized");
  }
 TORCH_CHECK(a.size()>0 && !(a.size() % 2),
             a.size()," pad size(s) given, expecting pairs for left,right or left,right,top,bottom.. etc");
 return o.pad(a.vec());
}

static void pad(bool a,K x,const PadImpl* m) {
 const torch::nn::functional::PadFuncOptions d({}), &o=m->options;
 OPTION(x, pad, klist(o.pad().size(),o.pad().data()));
 if(a || o.mode().index() != d.mode().index()) OPTION(x, mode,  ks(ESYM(o.mode())));
 if(a || o.value()        != d.value())        OPTION(x, value, kf(o.value()));
}

KAPI kpad(K x) {
 KTRY
  TORCH_CHECK(!x->t, "pad not implemented for ",kname(x->t));
  Tensor r, *t=xten(x,0);
  return kresult(t, torch::nn::functional::pad(t ? *t : kput(x,0), pad(x,1,Cast::pad)));
 KCATCH("pad");
}

// ----------------------------------------------------------------------------------
// cpad - constant pad w'fixed dimension and optional value (defaults to zero)
// ----------------------------------------------------------------------------------
template<size_t D,typename M> static M cpad(K x,J i,Cast c) {
 M o(0,0);
 bool sz=false; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
   switch(j) {
    case 0: o.padding(exarray<D*2>(x,i+j,c,Setting::pad)); sz=true; break;
    case 1: o.value(mdouble(x,i+j,c,Setting::value)); break;
    default: AT_ERROR(msym(c),": up to 2 positional args expected(padding;value), ",n," given");
  }
 while(xpair(p))
  switch(mset(p.k)) {
   case Setting::pad: o.padding(exarray<D*2>(p,c)); sz=true; break;
   case Setting::value: o.value(mdouble(p,c)); break;
   default: AT_ERROR("Unrecognized ",msym(c)," option: ",p.k); break;
  }
 TORCH_CHECK(sz, msym(c),": no padding sizes given");
 return o;
}

template<typename M> static void cpad(K x,const M* m) {
 OPTION(x, pad, KEX(m->options.padding()));
 OPTION(x, value, kf(m->options.value()));
}

// ----------------------------------------------------------------------------------
// npad - reflect/replicate/zero pad w'fixed dimension
// ----------------------------------------------------------------------------------
template<size_t D,typename M> static M npad(K x,J i,Cast c) {
 M o(0);
 bool sz=false; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
   switch(j) {
    case 0: o.padding(exarray<D*2>(x,i+j,c,Setting::pad)); sz=true; break;
    default: AT_ERROR(msym(c),": only 1 positional argument expected, ",n," given");
  }
 while(xpair(p))
  switch(mset(p.k)) {
   case Setting::pad: o.padding(exarray<D*2>(p,c)); sz=true; break;
   default: AT_ERROR("Unrecognized ",msym(c)," option: ",p.k); break;
  }
 TORCH_CHECK(sz, msym(c),": no padding sizes given");
 return o;
}

template<typename M> static void npad(K x,const M* m) {
 OPTION(x, pad, KEX(m->options.padding()));
}

// ------------------------------------------------------------------------------------
// noarg:  activation fns w'out args, logsigmoid,sigmoid,softsign,tanh,tanhshrink
// ------------------------------------------------------------------------------------
static void noarg(Cast c,K x,J i) {if(!xnone(x,i))AT_ERROR(msym(c),": no arguments expected");}

using Ft = Tensor (*)(const Tensor&);
static K noarg(const char* s,Ft f, K x) {
 KTRY
  Tensor *t=xten(x); return kresult(t, f(t ? *t : kput(x)));
 KCATCH(s);
}

KAPI gelu(K x)       {return noarg("gelu",       torch::gelu,                       x);}
KAPI logsigmoid(K x) {return noarg("logsigmoid", torch::log_sigmoid,                x);}
KAPI softsign(K x)   {return noarg("softsign",   torch::nn::functional::softsign,   x);}
KAPI tanhshrink(K x) {return noarg("tanhshrink", torch::nn::functional::tanhshrink, x);}

// ------------------------------------------------------------------------------------
// activation fns with inplace flag as only arg: relu,relu6,selu
// ------------------------------------------------------------------------------------
static bool inplace(K x,J i,Cast c) {
 bool b=false; Pairs p; J n=xargc(x,i,p);
 if(n)
  TORCH_CHECK(xbool(x,i,b) && n==1, msym(c),": unrecognized option(s), expecting single boolean flag");
 while(xpair(p)) {
  TORCH_CHECK(mset(p.k)==Setting::inplace, msym(c),": unrecognized option: ",p.k);
  b=mbool(p,c);
 }
 return b;
}

static void inplace(bool a,K x,bool b) {if(a || b) OPTION(x, inplace, kb(b));}

// ------------------------------------------------------------------------------------
//  elu,celu - exponential & continuously differentiable linear unit
//             accepts optional alpha & inplace flag
// ------------------------------------------------------------------------------------
template<typename O> static O alpha(K x,J i,Cast c) {
 O o; Pairs p; bool b; J n=xargc(x,i,p);
 if(n==1) {
  if(xbool(x,i,b))
    o.inplace(b);
  else
    o.alpha(mdouble(x,i,c,Setting::alpha));
 } else if(n==2) {
   o.alpha(mdouble(x,i,   c, Setting::alpha));
   o.inplace(mbool(x,i+1, c, Setting::inplace));
 } else if(n) {
  AT_ERROR(msym(c), ": unrecognized positional option(s), expecting alpha, inplace flag, or (alpha;inplace flag)");
 }
 while(xpair(p))
  switch(mset(p.k)) {
   case Setting::alpha:   o.alpha(mdouble(p,c)); break;
   case Setting::inplace: o.inplace(mbool(p,c)); break;
   default: AT_ERROR(msym(c)," option: ",p.k," not recognized");
  }
 return o;
}

template<typename O>static void alpha(bool a,Cast c,K x,const O& o) {
 O d;
 if(a || o.alpha()   != d.alpha())   OPTION(x, alpha,   kf(o.alpha()));
 if(a || o.inplace() != d.inplace()) OPTION(x, inplace, kb(o.inplace()));
}

// ------------------------------------------------------------------------------------
//  leakyrelu - allow a small positive gradient(slope) when x<0
// ------------------------------------------------------------------------------------
static torch::nn::LeakyReLUOptions slope(K x,J i,Cast c) {
 torch::nn::LeakyReLUOptions o; Pairs p; bool b; J n=xargc(x,i,p);
 if(n==1) {
  if(xbool(x,i,b))
    o.inplace(b);
  else
    o.negative_slope(mdouble(x,i,c,Setting::slope));
 } else if(n==2) {
   o.negative_slope(mdouble(x, i, c, Setting::slope));
   o.inplace(mbool(x,i+1, c, Setting::inplace));
 } else if(n) {
  AT_ERROR(msym(c), ": unrecognized positional option(s), expecting slope, inplace flag, or (slope;inplace flag)");
 }
 while(xpair(p))
  switch(mset(p.k)) {
   case Setting::slope:   o.negative_slope(mdouble(p,c)); break;
   case Setting::inplace: o.inplace(mbool(p,c)); break;
   default: AT_ERROR(msym(c)," option: ",p.k," not recognized");
  }
 return o;
}

static void slope(bool a,Cast c,K x,const torch::nn::LeakyReLUOptions& o) {
 torch::nn::LeakyReLUOptions d;
 if(a || o.negative_slope()   != d.negative_slope()) OPTION(x, slope,   kf(o.negative_slope()));
 if(a || o.inplace()          != d.inplace())        OPTION(x, inplace, kb(o.inplace()));
}

// ------------------------------------------------------------------------------------
// hardshrink, softshrink - module/function requires single parm: lambda
// ------------------------------------------------------------------------------------
static double lambda(Cast c) {
 return c==Cast::hardshrink ? torch::nn::HardshrinkOptions().lambda() 
                            : torch::nn::SoftshrinkOptions().lambda();
}

static double lambda(K x,J i,Cast c) {
 double l=lambda(c); Pairs p; J n=xargc(x,i,p);
 if(n==1) l=mdouble(x,i,c,Setting::lambda);
 TORCH_CHECK(n<2,msym(c),": unrecognized positional option(s), expecting lambda, e.g. 0.5");
 while(xpair(p)) {
  TORCH_CHECK(mset(p.k)==Setting::lambda,"Unrecognized option: ",p.k); l=mdouble(p,c);
 }
 return l;
}

static void lambda(bool a,Cast c,K x,double l) {if(a || l != lambda(c)) OPTION(x,lambda,kf(l));}

// ------------------------------------------------------------------------------------
// cat, glu & softmax,softmax,logsoftmax (modules only) accept single dimension arg
// ------------------------------------------------------------------------------------
static int64_t dim(Cast c) {
 switch(c) {
  case Cast::glu: return torch::nn::GLUOptions().dim();
  case Cast::cat: return CatOptions().dim();
  default:        return nj;
 }
}

static int64_t dim(K x,J i,Cast c) {
 int64_t d=dim(c); Pairs p; J n=xargc(x,i,p);
 if(n==1) d=int64(x,i,c,Setting::dim);
 TORCH_CHECK(n<2, msym(c),": unrecognized positional option(s), expecting single dimension");
 while(xpair(p)) {
  TORCH_CHECK(mset(p.k)==Setting::dim,"Unrecognized option: ",p.k); d=int64(p,c);
 }
 TORCH_CHECK(d!=nj, msym(c),": no dimension given");
 return d;
}

static void dim(bool a,Cast c,K x,int64_t d) {if(a || d != dim(c)) OPTION(x,dim,kj(d));}

// ----------------------------------------------------------------------------------
// softmax,softmin,logsoftmax: functional form requires dim & optional data type
// softdim: get default dimension from input tensor dimensions (deprecated)
// ----------------------------------------------------------------------------------
static J softdim(size_t d) {return !(d==0 || d==1 || d==3);}

static void softargs(K x,J i,Cast c,J &d,c10::optional<ScalarType>& s) { 
 s=c10::nullopt; Pairs p; J n=xargc(x,i,p);
 if(!((n==0 && p.n) || (xlong(x,i,d) && (n==1 || (n==2 && xtype(x,i+1,s))))))
  AT_ERROR(msym(c),": unrecognized arg(s), expecting dim or (dim;data type)");
 while(xpair(p))
  switch(mset(p.k)) {
   case Setting::dim:  d=plong(p); break;
   case Setting::type: s=ptype(p); break;
   default: AT_ERROR("Unrecognized ",msym(c)," option: ",p.k); break;
  }
 if(d==nj) 
  AT_ERROR("specify the dimension along which ",msym(c)," will be computed");
}

// -----------------------------------------------------------------------------------
// rrelu - randomized leaky relu, functional form has an additional flag for training
// -----------------------------------------------------------------------------------
static void rrelu(K x,J i,Cast c,bool fn,bool& tr,bool& in,double& lo,double& up) {
 Pairs p; J n=xargc(x,i,p); torch::nn::functional::RReLUFuncOptions o;
 lo=o.lower(); up=o.upper(); in=o.inplace(); tr=o.training();
 if(n) {
  if(fn) {
   TORCH_CHECK((n==1 && (xnum(x,i,lo) || xbool(x,i,tr))) ||
               (n==2 &&  xnum(x,i,lo) && (xnum(x,i+1,up) || xbool(x,i+1,tr))) ||
               (n==3 &&  xnum(x,i,lo) &&  xnum(x,i+1,up) && xbool(x,i+2,tr))  ||
               (n==4 &&  xnum(x,i,lo) &&  xnum(x,i+1,up) && xbool(x,i+2,tr) && xbool(x,i+3,in)),
               "rrelu: unexpected positional arg(s), expects (lower;upper;train flag;inplace flag)");
  } else {
   TORCH_CHECK((n==1 && (xnum(x,i,lo) || xbool(x,i,in))) ||
               (n==2 &&  xnum(x,i,lo) && (xnum(x,i+1,up) || xbool(x,i+1,in))) ||
               (n==3 &&  xnum(x,i,lo) &&  xnum(x,i+1,up) && xbool(x,i+2,in)),
               "rrelu: unexpected positional arg(s), expects (lower;upper;inplace flag)");
  }
 }
 while(xpair(p))
  switch(mset(p.k)) {
   case Setting::lower:   lo=mdouble(p,c); break;
   case Setting::upper:   up=mdouble(p,c); break;
   case Setting::train:   TORCH_CHECK(fn,"rrelu: training flag not set for module"); tr=mbool(p,c);   break;
   case Setting::inplace: in=mbool(p,c);   break;
   default: AT_ERROR("rrelu option: ",p.k," not recognized");
  }
}

// return options for rrelu module
static torch::nn::RReLUOptions rrelu(K x,J i,Cast c) {
 double lo,up; bool in,tr; rrelu(x,i,c,false,tr,in,lo,up);
 return torch::nn::RReLUOptions().lower(lo).upper(up).inplace(in);
}

// retrieve options from rrelu module
static void rrelu(bool a,K x,const torch::nn::RReLUOptions& o) {
 torch::nn::RReLUOptions d;
 if(a || d.lower()   != o.lower())   OPTION(x, lower,   kf(o.lower()));
 if(a || d.upper()   != o.upper())   OPTION(x, upper,   kf(o.upper()));
 if(a || d.inplace() != o.inplace()) OPTION(x, inplace, kb(o.inplace()));
}

// -----------------------------------------------------------------------------------------
// hardtanh - computationally cheaper version of tanh, straight line at min,max
// -----------------------------------------------------------------------------------------
static torch::nn::HardtanhOptions hardtanh(K x,J i,Cast c) {
 Pairs p; J n=xargc(x,i,p); torch::nn::HardtanhOptions o;
 bool b=o.inplace(); double v1=o.min_val(),v2=o.max_val();
 if(n) {
  TORCH_CHECK((n==1 && (xnum(x,i,v1) || xbool(x,i,b))) ||
              (n==2 &&  xnum(x,i,v1) && (xnum(x,i+1,v2) || xbool(x,i+1,b))) ||
              (n==3 &&  xnum(x,i,v1) &&  xnum(x,i+1,v2) && xbool(x,i+2,b)),
              "hardtanh: unexpected positional arg(s), expects (min;max;inplace flag)");
 }
 while(xpair(p))
  switch(mset(p.k)) {
   case Setting::min:     v1=mdouble(p,c); break;
   case Setting::max:     v2=mdouble(p,c); break;
   case Setting::inplace: b=mbool(p,c); break;
   default: AT_ERROR("hardtanh option: ",p.k," not recognized");
  }
 return o.min_val(v1).max_val(v2).inplace(b);
}

static void hardtanh(bool a,K x,const torch::nn::HardtanhOptions& o) {
 torch::nn::HardtanhOptions d;
 if(a || d.min_val() != o.min_val()) OPTION(x, min,     kf(o.min_val()));
 if(a || d.max_val() != o.max_val()) OPTION(x, max,     kf(o.max_val()));
 if(a || d.inplace() != o.inplace()) OPTION(x, inplace, kb(o.inplace()));
}

// -----------------------------------------------------------------------------------------
// softplus - smooth approximation to relu, can constrain to always be positive
// -----------------------------------------------------------------------------------------
static torch::nn::SoftplusOptions softplus(K x,J i,Cast c) {
 Pairs p; J n=xargc(x,i,p); torch::nn::SoftplusOptions o; double v1=o.beta(),v2=o.threshold();
 if(n) {
  TORCH_CHECK(xnum(x,i,v1) && (n==1 || (n==2 && xnum(x,i+1,v2))),
              "softplus: unexpected positional arg(s), expects (beta;threshold)");
 }
 while(xpair(p))
  switch(mset(p.k)) {
   case Setting::beta:      v1=mdouble(p,c); break;
   case Setting::threshold: v2=mdouble(p,c); break;
   default: AT_ERROR("softplus option: ",p.k," not recognized");
  }
 return o.beta(v1).threshold(v2);
}

static void softplus(bool a,K x,const torch::nn::SoftplusOptions& o) {
 torch::nn::SoftplusOptions d;
 if(a || d.beta()      != o.beta())      OPTION(x, beta,      kf(o.beta()));
 if(a || d.threshold() != o.threshold()) OPTION(x, threshold, kf(o.threshold()));
}

// ----------------------------------------------------------------------------------------------
// threshold - thresholds each element of input tensor, fns set/get threshold,value,inplace flag
// ----------------------------------------------------------------------------------------------
static torch::nn::ThresholdOptions threshold(K x,J i,Cast c) {
 Pairs p; J n=xargc(x,i,p); bool b=false; double v1=nf,v2=nf;
 if(n) {
  TORCH_CHECK((n==1 && (xnum(x,i,v1) || xbool(x,i,b))) ||
              (n==2 &&  xnum(x,i,v1) && (xnum(x,i+1,v2) || xbool(x,i+1,b))) ||
              (n==3 &&  xnum(x,i,v1) &&  xnum(x,i+1,v2) && xbool(x,i+2,b)),
              "threshold: unexpected positional arg(s), expects (threshold;value;inplace flag)");
 }
 while(xpair(p))
  switch(mset(p.k)) {
   case Setting::threshold: v1=mdouble(p,c); break;
   case Setting::value:     v2=mdouble(p,c); break;
   case Setting::inplace:   b=mbool(p,c); break;
   default: AT_ERROR("threshold option: ",p.k," not recognized");
  }
 TORCH_CHECK(v1 == v1 && v2 == v2, "threshold: both threshold level & replacement value must be given");
 return torch::nn::ThresholdOptions(v1,v2).inplace(b);
}

static void threshold(bool a,K x,const torch::nn::ThresholdOptions& o) {
 OPTION(x, threshold, kf(o.threshold()));
 OPTION(x, value,     kf(o.value()));
 if(a || o.inplace()) OPTION(x, inplace, kb(o.inplace()));
}

// -----------------------------------------------------------------------------------------
// functional form of activation fns:
//  relu,relu6,selu (inplace flag), elu,celu(alpha & inplace), leakyrelu(slope & inplace),
//  hardshrink,softshrink(lambda), glu(dim), rrelu(lower,upper & inplace flag)
//  hardtanh(min,max,inplace), softplus(beta,threshold), threshold(threshold,value,inplace)
// -----------------------------------------------------------------------------------------
static K act(K x,Cast c,const char* s) {
 KTRY
  bool a,p; Tensor r,t;
  if(xten(x,t))        p=true, a=false;
  else if(xten(x,0,t)) p=true, a=true;
  else if(xmixed(x,3)) p=false,a=true, t=kput(x,0);
  else                 p=false,a=false,t=kput(x);
  switch(c) {
   case Cast::relu:  r=torch::nn::functional::relu (t,a ? inplace(x,1,c) : false); break;
   case Cast::relu6: r=torch::nn::functional::relu6(t,a ? inplace(x,1,c) : false); break;
   case Cast::selu:  r=torch::nn::functional::selu (t,a ? inplace(x,1,c) : false); break;
   case Cast::elu:   r=torch::nn::functional::elu(t,alpha<torch::nn::ELUOptions>(a ? x : nullptr,1,c)); break;
   case Cast::celu:  r=torch::nn::functional::celu(t,alpha<torch::nn::CELUOptions>(a ? x : nullptr,1,c)); break;
   case Cast::leakyrelu: r=torch::nn::functional::leaky_relu(t,slope(a ? x : nullptr,1,c)); break;
   case Cast::hardshrink: r=torch::hardshrink(t,a ? lambda(x,1,c) : lambda(c)); break;
   case Cast::softshrink: r=torch::softshrink(t,a ? lambda(x,1,c) : lambda(c)); break;
   case Cast::glu:        r=torch::nn::functional::glu(t,a ? dim(x,1,c) : dim(c)); break;
   case Cast::softmin:
   case Cast::softmax:
   case Cast::logsoftmax: {
    auto d=softdim(t.dim()); c10::optional<ScalarType> s; if(a) softargs(x,1,c,d,s);
    switch(c) {
     case Cast::softmin:    r=torch::nn::functional::detail::softmin(t,d,s); break;
     case Cast::softmax:    r=torch::nn::functional::detail::softmax(t,d,s); break;
     case Cast::logsoftmax: r=torch::nn::functional::detail::log_softmax(t,d,s); break;
     default: AT_ERROR("Unrecognized activation function");
    }
    break;
   }
   case Cast::rrelu: {
    double lo,up; bool in,tr; rrelu(a ? x : nullptr,1,c,false,tr,in,lo,up);
    r=torch::nn::functional::detail::rrelu(t,lo,up,tr,in);
    break;
   }
   case Cast::hardtanh:  r=torch::nn::functional::hardtanh (t,  hardtanh(a ? x : nullptr,1,c)); break;
   case Cast::softplus:  r=torch::nn::functional::softplus (t,  softplus(a ? x : nullptr,1,c)); break;
   case Cast::threshold: r=torch::nn::functional::threshold(t, threshold(a ? x : nullptr,1,c)); break;
   default: AT_ERROR("Unrecognized activation function"); break;
  }
  return p && r.is_same(t) ? (K)0 : kresult(p,r);
 KCATCH(s);
}

KAPI       relu(K x) {return act(x, Cast::relu,       "relu");}
KAPI      relu6(K x) {return act(x, Cast::relu6,      "relu6");}
KAPI       selu(K x) {return act(x, Cast::selu,       "selu");}
KAPI        elu(K x) {return act(x, Cast::elu,        "elu");}
KAPI       celu(K x) {return act(x, Cast::celu,       "celu");}
KAPI  leakyrelu(K x) {return act(x, Cast::leakyrelu,  "leakyrelu");}
KAPI hardshrink(K x) {return act(x, Cast::hardshrink, "hardshrink");}
KAPI softshrink(K x) {return act(x, Cast::softshrink, "softshrink");}
KAPI        glu(K x) {return act(x, Cast::glu,        "glu");}
KAPI    softmin(K x) {return act(x, Cast::softmin,    "softmin");}
KAPI    softmax(K x) {return act(x, Cast::softmax,    "softmax");}
KAPI logsoftmax(K x) {return act(x, Cast::logsoftmax, "logsoftmax");}
KAPI      Rrelu(K x) {return act(x, Cast::rrelu,      "rrelu");}
KAPI   Hardtanh(K x) {return act(x, Cast::hardtanh,   "hardtanh");}
KAPI   Softplus(K x) {return act(x, Cast::softplus,   "softplus");}
KAPI  Threshold(K x) {return act(x, Cast::threshold,  "threshold");}

// -------------------------------------------------------------------------------------------
// prelu: parameterized relu
//        module accepts 1 or number of input parameters and optional initalization value
//        functional form requires weight directly rather than module's count & initial value
// -------------------------------------------------------------------------------------------
static torch::nn::PReLUOptions prelu(K x,J i,Cast c) {
 torch::nn::PReLUOptions o; auto m=o.num_parameters();auto w=o.init(); Pairs p; J n=xargc(x,i,p);
 if(n) TORCH_CHECK((n==1 && (xint64(x,i,m) || xdouble(x,i,w))) ||
                   (n==2 &&  xint64(x,i,m) && xdouble(x,i+1,w)),
                   "prelu: expecting 1-2 positional args in,init or (in;init)");
 while(xpair(p))
  switch(mset(p.k)) {
   case Setting::in:    m=int64(p,c); break;
   case Setting::init:  w=mdouble(p,c); break;
   default: AT_ERROR("prelu option: ",p.k," not recognized");
  }
 return o.num_parameters(m).init(w);
}

static void prelu(bool a,K x,const torch::nn::PReLUOptions& o) {
 torch::nn::PReLUOptions d;
 if(a || d.num_parameters() != o.num_parameters()) OPTION(x, in,   kj(o.num_parameters()));
 if(a || d.init()           != o.init())           OPTION(x, init, kf(o.init()));
}

KAPI Prelu(K x) {
 KTRY
  bool p; Tensor t,w;
  if(!x->t && x->n==2)
   p=xtenarg(x,t,w);
  else if(0<x->t && x->t<98 && x->n==2)
   p=false, t=kput(x), w=t[1], t=t[0];
  else
   AT_ERROR("prelu expects 2 args: input & weight, received ",kname(x->t),", count: ",xlen(x));
  return kresult(p, torch::prelu(t,w));
 KCATCH("prelu");
}

// ----------------------------------------------------------------------------------------------------
// distance funtions: could be considered layers or cost functions, so not declared static here
// similar - cosine similarity distance, parse/retrieve optional dimension and epsilon
// pairwise - pairwise distance, parse/retrieve optional power, eps, deep dimension flag
// ----------------------------------------------------------------------------------------------------
torch::nn::CosineSimilarityOptions similar(K x,J i,Cast c) {
 Pairs p; J n=xargc(x,i,p); torch::nn::CosineSimilarityOptions o;
 for(J j=0;j<n;++j)
  switch(j) {
   case 0: o.dim(int64(x,i+j,c,Setting::dim)); break;
   case 1: o.eps(mdouble(x,i+j,c,Setting::eps)); break;
   default: AT_ERROR(msym(c),": unrecognized positional arg(s), up to 2 args(dim,eps) expected, ",n," supplied");
  }
 while(xpair(p))
  switch(mset(p.k)) {
   case Setting::dim: o.dim(int64(p,c)); break;
   case Setting::eps: o.eps(mdouble(p,c)); break;
   default: AT_ERROR("Unrecognized option: ",p.k," for cosine similarity distance");
  }
 return o;
}

void similar(bool a,K x,const torch::nn::CosineSimilarityOptions& o) {
 torch::nn::CosineSimilarityOptions d; 
 if(a || (o.dim() != o.dim())) OPTION(x, dim, kj(o.dim()));
 if(a || (o.eps() != d.eps())) OPTION(x, eps, kf(o.eps()));
}

torch::nn::PairwiseDistanceOptions pairwise(K x,J i,Cast c) {
 Pairs p; J n=xargc(x,i,p); torch::nn::PairwiseDistanceOptions o;
 for(J j=0;j<n;++j)
  switch(j) {
   case 0: o.p(mdouble(x,i+j,c,Setting::p)); break;
   case 1: o.eps(mdouble(x,i+j,c,Setting::eps)); break;
   case 2: o.keepdim(mbool(x,i+j,c,Setting::keepdim)); break;
   default: AT_ERROR(msym(c),": unrecognized positional arg(s), up to 3 args(p,eps,keepdim) expected, ",n," supplied");
  }
 while(xpair(p))
  switch(mset(p.k)) {
   case Setting::p: o.p(mdouble(p,c)); break;
   case Setting::eps: o.eps(mdouble(p,c)); break;
   case Setting::keepdim: o.keepdim(mbool(p,c)); break;
   default: AT_ERROR("Unrecognized option: ",p.k," for pairwise distance");
  }
 return o;
}

void pairwise(bool a,K x,const torch::nn::PairwiseDistanceOptions& o) {
 torch::nn::PairwiseDistanceOptions d; 
 if(a || (o.p()       != d.p()))       OPTION(x, p,       kf(o.p()));
 if(a || (o.eps()     != d.eps()))     OPTION(x, eps,     kf(o.eps()));
 if(a || (o.keepdim() != d.keepdim())) OPTION(x, keepdim, kb(o.keepdim()));
}

// ------------------------------------------------------------------------
// functional form of the distance calculations
// ------------------------------------------------------------------------
static K distance(K x,Cast c) {
 KTRY
  TORCH_CHECK(!x->t, msym(c)," not implemented for ",kname(x->t));
  Tensor r, *a=xten(x,0), *b=xten(x,1);
  switch(c) {
   case Cast::pairwise: r=torch::nn::functional::pairwise_distance(a ? *a : kput(x,0), b ? *b : kput(x,1), pairwise(x,2,c)); break;
   case Cast::similar:  r=torch::nn::functional::cosine_similarity(a ? *a : kput(x,0), b ? *b : kput(x,1), similar(x,2,c)); break;
   default: AT_ERROR("Unrecognized distance function");
  }
  return kresult(a||b,r);
 KCATCH("distance");
}

KAPI Pairwise(K x) {return distance(x,Cast::pairwise);}
KAPI Similar(K x)  {return distance(x,Cast::similar);}

KAPI pdist(K x) {
 KTRY
  TORCH_CHECK(!x->t, "pdist not implemented for ",kname(x->t));
  F p=2; bool b=x->n==2 && xnum(x,1,p); Tensor *t = b ? xten(x,0) : xten(x);
  return kresult(t, torch::pdist(t ? *t : (b ? kput(x,0) : kput(x)), p));
 KCATCH("pdist");
}

// ----------------------------------------------------------------------------------------------------
// flatten - process arg(s) from k and return options
//         - return options used given a flatten module used
//         - call flatten as function given input/tensor and optional start & end dimensions
// ----------------------------------------------------------------------------------------------------
static torch::nn::FlattenOptions flatten(K x,J i) {
 torch::nn::FlattenOptions o; int64_t s=o.start_dim(),e=o.end_dim(); Pairs p; J n=xargc(x,i,p);
 if(!(n==0 || (xint64(x,i,s) && (n==1 || (n==2 && xint64(x,i+1,e))))))
  AT_ERROR("flatten: unrecognized arg(s)");
 while(xpair(p))
  switch(mset(p.k)) {
   case Setting::start: s=plong(p); break;
   case Setting::end:   e=plong(p); break;
   default: AT_ERROR("flatten option: ",p.k," not recognized");
  }
 return o.start_dim(s).end_dim(e);
}

static void flatten(bool a,K x,const torch::nn::FlattenImpl* m) {
 torch::nn::FlattenOptions d,o=m->options;
 if(a || d.start_dim() != o.start_dim()) OPTION(x, start, kj(o.start_dim()));
 if(a || d.end_dim()   != o.end_dim())   OPTION(x, end,   kj(o.end_dim()));
}

KAPI kflatten(K x) {
 KTRY
  bool m=false; Tensor t;
  auto o=flatten((xten(x,t) || xten(x,0,t) || (m=xmixed(x,3))) ? x : nullptr, 1);
  if(t.defined())
   return kten(torch::flatten(t, o.start_dim(), o.end_dim()));
  else
   return kget(torch::flatten(m ? kput(x,0) : kput(x), o.start_dim(), o.end_dim()));
 KCATCH("flatten");
}

// ----------------------------------------------------------------------------------------------------
// squeeze/unsqueeze - squeeze works with/without a dimension specified, unsqueeze requires it
// ----------------------------------------------------------------------------------------------------
static SqueezeOptions squeeze(K x,J i,Cast c) {
 SqueezeOptions o; Pairs p; bool b; J n=xargc(x,i,p);
 if(n==1) {
  if(xbool(x,i,b))
    o.inplace(b);
  else
    o.dim(int64n(x,i,c,Setting::dim));
 } else if(n==2) {
   o.dim(   int64n(x,i,   c, Setting::dim));
   o.inplace(mbool(x,i+1, c, Setting::inplace));
 } else if(n) {
  AT_ERROR(msym(c), ": unrecognized positional arg(s), expecting dim, inplace flag, or (dim;inplace flag)");
 }
 while(xpair(p))
  switch(mset(p.k)) {
   case Setting::dim:     o.dim(int64n(p,c)); break;
   case Setting::inplace: o.inplace(mbool(p,c)); break;
   default: AT_ERROR(msym(c)," option: ",p.k," not recognized");
  }
 TORCH_CHECK(c==Cast::squeeze || o.dim().has_value(), msym(c),": no dimension given");
 return o;
}

static void squeeze(bool a,K x,const SqueezeOptions& o) {
 if(o.dim().has_value()) OPTION(x, dim,     kj(o.dim().value()));
 if(a || o.inplace())    OPTION(x, inplace, kb(o.inplace()));
}

// ----------------------------------------------------------------------------------------------------
// getsize - get size(s) for expand & reshape
// expand
// reshape
// ----------------------------------------------------------------------------------------------------
static SizeOptions getsize(K x,J i,Cast c) {
 IntArrayRef a; LongVector v; Pairs p; J n=xargc(x,i,p);
 TORCH_CHECK(!n || (xsize(x,i,a) && n==1), msym(c)," expects size(s) as argument");
 while(xpair(p))
  switch(mset(p.k)) {
   case Setting::size: psize(p,a); break;
   default: AT_ERROR(msym(c)," option: ",p.k," not recognized");
  }
 for(auto j:a) v.push_back(j);
 return SizeOptions(v);
}

static void getsize(bool a,K x,const SizeOptions& o) {
 OPTION(x, size, klist(o.size().size(),o.size().data()));
}

// ----------------------------------------------------------------------------------------------------
// anymodule - define module from supplied options, return in generic AnyModule container
// ----------------------------------------------------------------------------------------------------
AnyModule anymodule(K x,J i,Cast c) {
 switch(c) {
  case Cast::batchnorm1d:  return AnyModule(torch::nn::BatchNorm1d(batchnorm<torch::nn::BatchNormOptions>(x,i,c)));
  case Cast::batchnorm2d:  return AnyModule(torch::nn::BatchNorm2d(batchnorm<torch::nn::BatchNormOptions>(x,i,c)));
  case Cast::batchnorm3d:  return AnyModule(torch::nn::BatchNorm3d(batchnorm<torch::nn::BatchNormOptions>(x,i,c)));

  case Cast::instancenorm1d:  return AnyModule(torch::nn::InstanceNorm1d(batchnorm<torch::nn::InstanceNormOptions>(x,i,c)));
  case Cast::instancenorm2d:  return AnyModule(torch::nn::InstanceNorm2d(batchnorm<torch::nn::InstanceNormOptions>(x,i,c)));
  case Cast::instancenorm3d:  return AnyModule(torch::nn::InstanceNorm3d(batchnorm<torch::nn::InstanceNormOptions>(x,i,c)));

  case Cast::groupnorm:  return AnyModule(torch::nn::GroupNorm(groupnorm(x,i,c)));
  case Cast::layernorm:  return AnyModule(torch::nn::LayerNorm(layernorm(x,i,c)));
  case Cast::localnorm:  return AnyModule(torch::nn::LocalResponseNorm(localnorm<torch::nn::LocalResponseNormOptions>(x,i,c)));
  case Cast::crossmap2d: return AnyModule(torch::nn::CrossMapLRN2d(localnorm<torch::nn::CrossMapLRN2dOptions>(x,i,c)));

  case Cast::embed:        return AnyModule(embed(x,i,c));
  case Cast::embedbag:     return AnyModule(embedbag(x,i,c));
  case Cast::linear:       return AnyModule(torch::nn::Linear(linear(x,i,c)));
  case Cast::bilinear:     return AnyModule(torch::nn::Bilinear(bilinear(x,i,c)));

  case Cast::drop:         return AnyModule(torch::nn::Dropout(drop(x,i,c)));
  case Cast::drop2d:       return AnyModule(torch::nn::Dropout2d(drop(x,i,c)));
  case Cast::drop3d:       return AnyModule(torch::nn::Dropout3d(drop(x,i,c)));
  case Cast::adrop:        return AnyModule(torch::nn::AlphaDropout(drop(x,i,c)));
  case Cast::fadrop:       return AnyModule(torch::nn::FeatureAlphaDropout(drop(x,i,c)));

  case Cast::conv1d:       return AnyModule(torch::nn::Conv1d(conv<1>(x,i,c)));
  case Cast::conv2d:       return AnyModule(torch::nn::Conv2d(conv<2>(x,i,c)));
  case Cast::conv3d:       return AnyModule(torch::nn::Conv3d(conv<3>(x,i,c)));

  case Cast::convtranspose1d:  return AnyModule(ConvTranspose1d(convtran<1>(x,i,c)));
  case Cast::convtranspose2d:  return AnyModule(ConvTranspose2d(convtran<2>(x,i,c)));
  case Cast::convtranspose3d:  return AnyModule(ConvTranspose3d(convtran<3>(x,i,c)));

  case Cast::fold:         return AnyModule(torch::nn::Fold(fold(x,i,c)));
  case Cast::unfold:       return AnyModule(torch::nn::Unfold(unfold(x,i,c)));

  case Cast::maxpool1d:    return AnyModule(torch::nn::MaxPool1d(maxpool<1>(x,i,c)));
  case Cast::maxpool2d:    return AnyModule(torch::nn::MaxPool2d(maxpool<2>(x,i,c)));
  case Cast::maxpool3d:    return AnyModule(torch::nn::MaxPool3d(maxpool<3>(x,i,c)));

  case Cast::avgpool1d:    return AnyModule(torch::nn::AvgPool1d(avgpool<1>(x,i,c)));
  case Cast::avgpool2d:    return AnyModule(torch::nn::AvgPool2d(avgpool<2>(x,i,c)));
  case Cast::avgpool3d:    return AnyModule(torch::nn::AvgPool3d(avgpool<3>(x,i,c)));

  case Cast::adaptmax1d:   return AnyModule(torch::nn::AdaptiveMaxPool1d(adapt<1,torch::nn::AdaptiveMaxPool1dOptions>(x,i,c)));
  case Cast::adaptmax2d:   return AnyModule(torch::nn::AdaptiveMaxPool2d(adapt<2,torch::nn::AdaptiveMaxPool2dOptions>(x,i,c)));
  case Cast::adaptmax3d:   return AnyModule(torch::nn::AdaptiveMaxPool3d(adapt<3,torch::nn::AdaptiveMaxPool3dOptions>(x,i,c)));

  case Cast::adaptavg1d:   return AnyModule(torch::nn::AdaptiveAvgPool1d(adapt<1,torch::nn::AdaptiveAvgPool1dOptions>(x,i,c)));
  case Cast::adaptavg2d:   return AnyModule(torch::nn::AdaptiveAvgPool2d(adapt<2,torch::nn::AdaptiveAvgPool2dOptions>(x,i,c)));
  case Cast::adaptavg3d:   return AnyModule(torch::nn::AdaptiveAvgPool3d(adapt<3,torch::nn::AdaptiveAvgPool3dOptions>(x,i,c)));

  case Cast::fmaxpool2d:   return AnyModule(torch::nn::FractionalMaxPool2d(fpool<2>(x,i,c)));
  case Cast::fmaxpool3d:   return AnyModule(torch::nn::FractionalMaxPool3d(fpool<3>(x,i,c)));

  case Cast::lppool1d:     return AnyModule(torch::nn::LPPool1d(lppool<1>(x,i,c)));
  case Cast::lppool2d:     return AnyModule(torch::nn::LPPool2d(lppool<2>(x,i,c)));

  case Cast::pad:          return AnyModule(Pad(pad(x,i,c)));
  case Cast::pad1d:        return AnyModule(torch::nn::ConstantPad1d(cpad<1,torch::nn::ConstantPad1dOptions>(x,i,c)));
  case Cast::pad2d:        return AnyModule(torch::nn::ConstantPad2d(cpad<2,torch::nn::ConstantPad2dOptions>(x,i,c)));
  case Cast::pad3d:        return AnyModule(torch::nn::ConstantPad3d(cpad<3,torch::nn::ConstantPad3dOptions>(x,i,c)));
  case Cast::reflect1d:    return AnyModule(torch::nn::ReflectionPad1d(npad<1,torch::nn::ReflectionPad1dOptions>(x,i,c)));
  case Cast::reflect2d:    return AnyModule(torch::nn::ReflectionPad2d(npad<2,torch::nn::ReflectionPad2dOptions>(x,i,c)));
  case Cast::replicate1d:  return AnyModule(torch::nn::ReplicationPad1d(npad<1,torch::nn::ReplicationPad1dOptions>(x,i,c)));
  case Cast::replicate2d:  return AnyModule(torch::nn::ReplicationPad2d(npad<2,torch::nn::ReplicationPad2dOptions>(x,i,c)));
  case Cast::replicate3d:  return AnyModule(torch::nn::ReplicationPad3d(npad<3,torch::nn::ReplicationPad3dOptions>(x,i,c)));
  case Cast::zeropad2d:    return AnyModule(torch::nn::ZeroPad2d(npad<2,torch::nn::ZeroPad2dOptions>(x,i,c)));

  case Cast::rnn:          return AnyModule((rnn<torch::nn::RNN, torch::nn::RNNOptions> (c,x,i)));
  case Cast::gru:          return AnyModule((rnn<torch::nn::GRU, torch::nn::GRUOptions> (c,x,i)));
  case Cast::lstm:         return AnyModule((rnn<torch::nn::LSTM,torch::nn::LSTMOptions>(c,x,i)));

  case Cast::identity:     noarg(c,x,i); return AnyModule(torch::nn::Identity());
  case Cast::logsigmoid:   noarg(c,x,i); return AnyModule(torch::nn::LogSigmoid());
  case Cast::sigmoid:      noarg(c,x,i); return AnyModule(torch::nn::Sigmoid());
  case Cast::softsign:     noarg(c,x,i); return AnyModule(torch::nn::Softsign());
  case Cast::softmax2d:    noarg(c,x,i); return AnyModule(torch::nn::Softmax2d());
  case Cast::tanh:         noarg(c,x,i); return AnyModule(torch::nn::Tanh());
  case Cast::tanhshrink:   noarg(c,x,i); return AnyModule(torch::nn::Tanhshrink());
  case Cast::gelu:         noarg(c,x,i); return AnyModule(torch::nn::GELU());

  case Cast::relu:         return AnyModule( torch::nn::ReLU(inplace(x,i,c)));
  case Cast::relu6:        return AnyModule(torch::nn::ReLU6(inplace(x,i,c)));
  case Cast::selu:         return AnyModule( torch::nn::SELU(inplace(x,i,c)));

  case Cast::softmax:      return AnyModule(torch::nn::Softmax(dim(x,i,c)));
  case Cast::softmin:      return AnyModule(torch::nn::Softmin(dim(x,i,c)));
  case Cast::logsoftmax:   return AnyModule(torch::nn::LogSoftmax(dim(x,i,c)));
  case Cast::flatten:      return AnyModule(torch::nn::Flatten(flatten(x,i)));

  case Cast::squeeze:      return AnyModule(Squeeze(squeeze(x,i,c)));
  case Cast::unsqueeze:    return AnyModule(Unsqueeze(squeeze(x,i,c)));
  case Cast::expand:       return AnyModule(Expand(getsize(x,i,c)));
  case Cast::reshape:      return AnyModule(Reshape(getsize(x,i,c)));
  case Cast::cat:          return AnyModule(Cat(dim(x,i,c)));

  case Cast::elu:          return AnyModule(torch::nn::ELU (alpha<torch::nn::ELUOptions> (x,i,c)));
  case Cast::celu:         return AnyModule(torch::nn::CELU(alpha<torch::nn::CELUOptions>(x,i,c)));
  case Cast::leakyrelu:    return AnyModule(torch::nn::LeakyReLU(slope(x,i,c)));
  case Cast::glu:          return AnyModule(torch::nn::GLU(dim(x,i,c)));
  case Cast::hardshrink:   return AnyModule(torch::nn::Hardshrink(lambda(x,i,c)));
  case Cast::softshrink:   return AnyModule(torch::nn::Softshrink(lambda(x,i,c)));
  case Cast::prelu:        return AnyModule(torch::nn::PReLU(prelu(x,i,c)));
  case Cast::rrelu:        return AnyModule(torch::nn::RReLU(rrelu(x,i,c)));
  case Cast::hardtanh:     return AnyModule(torch::nn::Hardtanh(hardtanh(x,i,c)));
  case Cast::softplus:     return AnyModule(torch::nn::Softplus(softplus(x,i,c)));
  case Cast::threshold:    return AnyModule(torch::nn::Threshold(threshold(x,i,c)));

  case Cast::pairwise:     return AnyModule(torch::nn::PairwiseDistance(pairwise(x,i,c)));
  case Cast::similar:      return AnyModule(torch::nn::CosineSimilarity(similar(x,i,c)));
  default: AT_ERROR("Unrecognized module: ",(I)c);
 }
}

// ----------------------------------------------------------------------------------------------------
// mparms - set parameters/buffers in a defined module from k values in dictionary with matching names
// pushback - define modules, reset parameter/buffer values from a previous state, add to sequential
// ----------------------------------------------------------------------------------------------------
static void mparms(S s,Module &m,K x,bool p) { // set named parms/buffers in module m from dict x, p true if parms
 K k=kK(x)[0],v=kK(x)[1]; Tensor V; if(v->t) V=kput(v);
 for(auto &a:p ? m.named_parameters() : m.named_buffers()) {
  J i=kfind(k,a.key());
  if(i<0) {
   AT_ERROR("Unable to find ",s,(p ? " parameter" : " buffer"),": ",a.key());
   break;
  }
  Tensor t=v->t ? V[i] : kput(kK(v)[i]);
  if(a.value().defined()) {
   torch::NoGradGuard g;
   if(a.value().dtype() != t.dtype())
    AT_ERROR("Type mismatch: ",s,(p ? " parameter " : " buffer "),a.key()," is ",a.value().dtype(),", input is ",t.dtype());
   if(!a.value().is_same_size(t))
    AT_ERROR("Size mismatch: ",s,(p ? " parameter " : " buffer "),a.key()," is ",a.value().sizes(),", input is ",t.sizes());
   if (a.value().device() != t.device())
    a.value().set_data(t);
   else
    a.value().set_(t);
  } else {
   a.value()=std::move(t);
  }
 }
}

void pushback(Sequential &q,S s,S n=nullptr,J i=-1,K x=nullptr,K p=nullptr,K f=nullptr);
void pushback(Sequential &q,S s,S n,        J i,   K x,        K p,        K f)  {
 Cast c=msym(s);
 auto m=anymodule(x,i,c);           // define module, return in generic container
 if(p) mparms(s,*m.ptr(),p,true);   // add parameter values if supplied
 if(f) mparms(s,*m.ptr(),f,false);  // add any buffers supplied
 n ? q->push_back(std::string(n),m) : q->push_back(m);
}

void pushback(Sequential &q,K x) { // add modules to sequential from k table of options or full state
 J n=x->t==99 ? 0 : xlen(x);
 for(J i=98-x->t;i<n;++i)
   pushback(q,statemodule(x,i),statename(x,i),-1,stateoptions(x,i),stateparms(x,i),statebuffers(x,i));
}

// --------------------------------------------------------------------------------------------
//  functions to extract module settings and state -> q dictionary/table
// --------------------------------------------------------------------------------------------
// mopt - given module, cast at runtime to known type and extract options as k dictionary
// mget - extract module options and, optionally, parameters & buffers to k array
// mtable - extract child modules and return as k table, one row per module
// --------------------------------------------------------------------------------------------
std::tuple<Cast,K> mopt(bool a,const Module& g) { //a:all options returned if true, else only non-default
 Cast c=Cast::undefined; K x=xD(ktn(KS,0),ktn(0,0));
 if       (auto* m=g.as<torch::nn::Sequential>())        { c=Cast::sequential;
 } else if(auto* m=g.as<Join>())                         { c=Cast::join;

 } else if(auto* m=g.as<torch::nn::BatchNorm1d>())       { c=Cast::batchnorm1d;    batchnorm(a,x,m->options);
 } else if(auto* m=g.as<torch::nn::BatchNorm2d>())       { c=Cast::batchnorm2d;    batchnorm(a,x,m->options);
 } else if(auto* m=g.as<torch::nn::BatchNorm3d>())       { c=Cast::batchnorm3d;    batchnorm(a,x,m->options);
 } else if(auto* m=g.as<torch::nn::InstanceNorm1d>())    { c=Cast::instancenorm1d; batchnorm(a,x,m->options);
 } else if(auto* m=g.as<torch::nn::InstanceNorm2d>())    { c=Cast::instancenorm2d; batchnorm(a,x,m->options);
 } else if(auto* m=g.as<torch::nn::InstanceNorm3d>())    { c=Cast::instancenorm3d; batchnorm(a,x,m->options);
 } else if(auto* m=g.as<torch::nn::GroupNorm>())         { c=Cast::groupnorm;      groupnorm(a,x,m->options);
 } else if(auto* m=g.as<torch::nn::LayerNorm>())         { c=Cast::layernorm;      layernorm(a,x,m->options);
 } else if(auto* m=g.as<torch::nn::LocalResponseNorm>()) { c=Cast::localnorm;      localnorm(a,x,c,m->options);
 } else if(auto* m=g.as<torch::nn::CrossMapLRN2d>())     { c=Cast::crossmap2d;     localnorm(a,x,c,m->options);

 } else if(auto* m=g.as<torch::nn::Embedding>())         { c=Cast::embed;    embed(a,x,c,m->options,m->weight);
 } else if(auto* m=g.as<torch::nn::EmbeddingBag>())      { c=Cast::embedbag; embed(a,x,c,m->options,m->weight);
 } else if(auto* m=g.as<torch::nn::Linear>())            { c=Cast::linear;   linear(a,x,m);
 } else if(auto* m=g.as<torch::nn::Bilinear>())          { c=Cast::bilinear; bilinear(a,x,m);

 } else if(auto* m=g.as<torch::nn::Dropout>())             { c=Cast::drop;   drop(a,x,m->options);
 } else if(auto* m=g.as<torch::nn::Dropout2d>())           { c=Cast::drop2d; drop(a,x,m->options);
 } else if(auto* m=g.as<torch::nn::Dropout3d>())           { c=Cast::drop3d; drop(a,x,m->options);
 } else if(auto* m=g.as<torch::nn::AlphaDropout>())        { c=Cast::adrop;  drop(a,x,m->options);
 } else if(auto* m=g.as<torch::nn::FeatureAlphaDropout>()) { c=Cast::fadrop; drop(a,x,m->options);

 } else if(auto* m=g.as<torch::nn::Conv1d>())         { c=Cast::conv1d; conv<1>(a,x,m->options);
 } else if(auto* m=g.as<torch::nn::Conv2d>())         { c=Cast::conv2d; conv<2>(a,x,m->options);
 } else if(auto* m=g.as<torch::nn::Conv3d>())         { c=Cast::conv3d; conv<3>(a,x,m->options);
 } else if(auto* m=g.as<torch::nn::ConvTranspose1d>()){ c=Cast::convtranspose1d; conv<1>(a,x,m->options);
 } else if(auto* m=g.as<torch::nn::ConvTranspose2d>()){ c=Cast::convtranspose2d; conv<2>(a,x,m->options);
 } else if(auto* m=g.as<torch::nn::ConvTranspose3d>()){ c=Cast::convtranspose3d; conv<3>(a,x,m->options);

 } else if(auto* m=g.as<torch::nn::Fold>())           { c=Cast::fold;     fold(a,x,m->options);
 } else if(auto* m=g.as<torch::nn::Unfold>())         { c=Cast::unfold; unfold(a,x,m->options);

 } else if(auto* m=g.as<torch::nn::MaxPool1d>())      { c=Cast::maxpool1d; maxpool<1,torch::nn::MaxPool1dImpl>(a,x,m);
 } else if(auto* m=g.as<torch::nn::MaxPool2d>())      { c=Cast::maxpool2d; maxpool<2,torch::nn::MaxPool2dImpl>(a,x,m);
 } else if(auto* m=g.as<torch::nn::MaxPool3d>())      { c=Cast::maxpool3d; maxpool<3,torch::nn::MaxPool3dImpl>(a,x,m);

 } else if(auto* m=g.as<torch::nn::AvgPool1d>())      { c=Cast::avgpool1d; avgpool<1,torch::nn::AvgPool1dImpl>(a,x,m);
 } else if(auto* m=g.as<torch::nn::AvgPool2d>())      { c=Cast::avgpool2d; avgpool<2,torch::nn::AvgPool2dImpl>(a,x,m);
 } else if(auto* m=g.as<torch::nn::AvgPool3d>())      { c=Cast::avgpool3d; avgpool<3,torch::nn::AvgPool3dImpl>(a,x,m);

 } else if(auto* m=g.as<torch::nn::AdaptiveMaxPool1d>())   { c=Cast::adaptmax1d; adapt<torch::nn::AdaptiveMaxPool1dImpl>(x,m);
 } else if(auto* m=g.as<torch::nn::AdaptiveMaxPool2d>())   { c=Cast::adaptmax2d; adapt<torch::nn::AdaptiveMaxPool2dImpl>(x,m);
 } else if(auto* m=g.as<torch::nn::AdaptiveMaxPool3d>())   { c=Cast::adaptmax3d; adapt<torch::nn::AdaptiveMaxPool3dImpl>(x,m);

 } else if(auto* m=g.as<torch::nn::AdaptiveAvgPool1d>())   { c=Cast::adaptmax1d; adapt<torch::nn::AdaptiveAvgPool1dImpl>(x,m);
 } else if(auto* m=g.as<torch::nn::AdaptiveAvgPool2d>())   { c=Cast::adaptmax2d; adapt<torch::nn::AdaptiveAvgPool2dImpl>(x,m);
 } else if(auto* m=g.as<torch::nn::AdaptiveAvgPool3d>())   { c=Cast::adaptmax3d; adapt<torch::nn::AdaptiveAvgPool3dImpl>(x,m);

 } else if(auto* m=g.as<torch::nn::FractionalMaxPool2d>()) { c=Cast::fmaxpool2d; fpool<2,torch::nn::FractionalMaxPool2dImpl>(a,x,m);
 } else if(auto* m=g.as<torch::nn::FractionalMaxPool3d>()) { c=Cast::fmaxpool3d; fpool<3,torch::nn::FractionalMaxPool3dImpl>(a,x,m);

 } else if(auto* m=g.as<torch::nn::LPPool1d>())         { c=Cast::lppool1d; lppool<1,torch::nn::LPPool1dImpl>(a,x,m);
 } else if(auto* m=g.as<torch::nn::LPPool2d>())         { c=Cast::lppool2d; lppool<2,torch::nn::LPPool2dImpl>(a,x,m);

 } else if(auto* m=g.as<Pad>())                         { c=Cast::pad;         pad(a,x,m);
 } else if(auto* m=g.as<torch::nn::ConstantPad1d>())    { c=Cast::pad1d;       cpad(x,m);
 } else if(auto* m=g.as<torch::nn::ConstantPad2d>())    { c=Cast::pad2d;       cpad(x,m);
 } else if(auto* m=g.as<torch::nn::ConstantPad3d>())    { c=Cast::pad3d;       cpad(x,m);
 } else if(auto* m=g.as<torch::nn::ReflectionPad1d>())  { c=Cast::reflect1d;   npad(x,m);
 } else if(auto* m=g.as<torch::nn::ReflectionPad2d>())  { c=Cast::reflect2d;   npad(x,m);
 } else if(auto* m=g.as<torch::nn::ReplicationPad1d>()) { c=Cast::replicate1d; npad(x,m);
 } else if(auto* m=g.as<torch::nn::ReplicationPad2d>()) { c=Cast::replicate2d; npad(x,m);
 } else if(auto* m=g.as<torch::nn::ReplicationPad3d>()) { c=Cast::replicate3d; npad(x,m);
 } else if(auto* m=g.as<torch::nn::ZeroPad2d>())        { c=Cast::zeropad2d;   npad(x,m);

 } else if(auto* m=g.as<torch::nn::RNN>())   { c=Cast::rnn;  rnn<torch::nn::RNNImpl,  torch::nn::RNNOptions> (a,x,m);
 } else if(auto* m=g.as<torch::nn::GRU>())   { c=Cast::gru;  rnn<torch::nn::GRUImpl,  torch::nn::GRUOptions> (a,x,m);
 } else if(auto* m=g.as<torch::nn::LSTM>())  { c=Cast::lstm; rnn<torch::nn::LSTMImpl, torch::nn::LSTMOptions>(a,x,m);

 } else if(g.as<torch::nn::Identity>())      { c=Cast::identity;
 } else if(g.as<torch::nn::LogSigmoid>())    { c=Cast::logsigmoid;
 } else if(g.as<torch::nn::Sigmoid>())       { c=Cast::sigmoid;
 } else if(g.as<torch::nn::Softsign>())      { c=Cast::softsign;
 } else if(g.as<torch::nn::Softmax2d>())     { c=Cast::softmax2d;
 } else if(g.as<torch::nn::Tanh>())          { c=Cast::tanh;
 } else if(g.as<torch::nn::Tanhshrink>())    { c=Cast::tanhshrink;
 } else if(g.as<torch::nn::GELU>())          { c=Cast::gelu;

 } else if(auto* m=g.as<torch::nn::ReLU>())  { c=Cast::relu;  inplace(a,x,m->options.inplace());
 } else if(auto* m=g.as<torch::nn::SELU>())  { c=Cast::selu;  inplace(a,x,m->options.inplace());
 } else if(auto* m=g.as<torch::nn::ReLU6>()) { c=Cast::relu6; inplace(a,x,m->options.inplace());

 } else if(auto* m=g.as<torch::nn::Softmax>())    { c=Cast::softmax;    OPTION(x, dim, kj(m->options.dim()));
 } else if(auto* m=g.as<torch::nn::Softmin>())    { c=Cast::softmin;    OPTION(x, dim, kj(m->options.dim()));
 } else if(auto* m=g.as<torch::nn::LogSoftmax>()) { c=Cast::logsoftmax; OPTION(x, dim, kj(m->options.dim()));
 } else if(auto* m=g.as<torch::nn::Flatten>())    { c=Cast::flatten;    flatten(a,x,m);

 } else if(auto* m=g.as<Squeeze>())    { c=Cast::squeeze;    squeeze(a,x,m->options);
 } else if(auto* m=g.as<Unsqueeze>())  { c=Cast::unsqueeze;  squeeze(a,x,m->options);
 } else if(auto* m=g.as<Expand>())     { c=Cast::expand;     getsize(a,x,m->options);
 } else if(auto* m=g.as<Reshape>())    { c=Cast::reshape;    getsize(a,x,m->options);
 } else if(auto* m=g.as<Cat>())        { c=Cast::cat;        dim(a,c,x,m->options.dim());

 } else if(auto* m=g.as<torch::nn::ELU>())        { c=Cast::elu;  alpha(a,c,x,m->options);
 } else if(auto* m=g.as<torch::nn::CELU>())       { c=Cast::celu; alpha(a,c,x,m->options);
 } else if(auto* m=g.as<torch::nn::LeakyReLU>())  { c=Cast::leakyrelu;  slope(a,c,x,m->options);
 } else if(auto* m=g.as<torch::nn::GLU>())        { c=Cast::glu;        dim(a,c,x,m->options.dim());
 } else if(auto* m=g.as<torch::nn::Hardshrink>()) { c=Cast::hardshrink; lambda(a,c,x,m->options.lambda());
 } else if(auto* m=g.as<torch::nn::Softshrink>()) { c=Cast::softshrink; lambda(a,c,x,m->options.lambda());

 } else if(auto* m=g.as<torch::nn::PReLU>())      { c=Cast::prelu;      prelu(a,x,m->options);
 } else if(auto* m=g.as<torch::nn::RReLU>())      { c=Cast::rrelu;      rrelu(a,x,m->options);
 } else if(auto* m=g.as<torch::nn::Hardtanh>())   { c=Cast::hardtanh;   hardtanh(a,x,m->options);
 } else if(auto* m=g.as<torch::nn::Softplus>())   { c=Cast::softplus;   softplus(a,x,m->options);
 } else if(auto* m=g.as<torch::nn::Threshold>())  { c=Cast::threshold;  threshold(a,x,m->options);

 } else if(auto* m=g.as<torch::nn::PairwiseDistance>())  { c=Cast::pairwise; pairwise(a,x,m->options);
 } else if(auto* m=g.as<torch::nn::CosineSimilarity>())  { c=Cast::similar;  similar(a,x,m->options);
 } else { AT_ERROR("Unrecognized module: ",g.name());
 }
 return std::make_tuple(c,x);
}

void mget(bool a,int64_t d,const char* s,bool t,const Module& m,K x) {
 Cast c; K o,*k=kK(x); std::tie(c,o)=mopt(a,m);
 if(t) {
  ja(&k[0], &d);
  js(&k[1], msym(c));
  js(&k[2], cs(s));
  jk(&k[3], o);
  if(x->n == 6)
   jk(&k[4], kdict(m.named_parameters(false))),
   jk(&k[5], kdict(m.named_buffers(false)));
  for(auto& i:m.named_children())
   mget(a,d+1,i.key().c_str(),t,*i.value(),x);
 } else {
  TORCH_CHECK(!m.children().size(), msym(c), ": unexpected child module(s)");
  k[0]=kj(d);
  k[1]=ks(msym(c));
  k[2]=ks(cs(s));
  k[3]=o;
  if(x->n == 6)
   k[4]=kdict(m.named_parameters(false)),
   k[5]=kdict(m.named_buffers(false));
 }
}

K mget(bool a,bool b,const char* s,const Module& m) {
 K k=mkeys(b),v=ktn( 0, b ? 6 : 4);  // key,val for depth,module,name,options w'parms & buffers if b
 if(container(m)) {
  for(J i=0; i<v->n; ++i) kK(v)[i]=ktn(!i ? KJ : (i<3 ? KS : 0), 0);
  mget(a,0,s,true,m,v);
  return xT(xD(k,v));
 } else {
  mget(a,0,s,false,m,v);
  return xD(k,v);
 }
}

K mtable(const Sequential& q,bool a,bool b) {
 return mget(a,b,"",*q);
}

// --------------------------------------------------------------------------------------
// tchild - extract named parameter/buffer tensor from child module in sequential
// mchild - extract child module by index/name, return module state or individual tensor
// --------------------------------------------------------------------------------------
static K tchild(S s,const Module& c) {
 Tensor t;
 if(c.named_parameters().contains(s))
  t=c.named_parameters()[s];
 else if(c.named_buffers().contains(s))
  t=c.named_buffers()[s];
 else
  AT_ERROR("No parameter or buffer named: ",s);
 return kget(t);
}

static K mchild(bool a,J i,S s,const Sequential &q) {
 if(i<0 || (unsigned)i>=q->size())
  AT_ERROR("Invalid module index: ",i);
 if(s) {
  return tchild(s,*q->children()[i]);
 } else {
  //direct access by index[0] fails to pick up name(?)
  //const auto& c=q->named_children()[i];
  //mget(c.key().c_str(),*c.value(),a,v,-1);
  const auto& c=q->named_children();
  return mget(a,true,c.keys()[i].c_str(),*c.values()[i]);
 }
}

static K mchild(bool a,S s1,S s2,const Sequential &q) {
 const auto& m=q->named_children()[s1];
 if(s2) {
  return tchild(s2,*m);
 } else {
  return mget(a,true,s1,*m);
 }
}

// ------------------------------------------------------------------------------------------
//  main api functions defined in k
// ------------------------------------------------------------------------------------------
// margs - helper function used to parse module creation args (if not table/dictionary)
// seq - create/append sequential module
// mstate - class,module,name,options,parms,buffers for module(s) or selected parm/buffer
// seqforward - given sequential module and input, run forward calcs, return tensor to k
// ------------------------------------------------------------------------------------------
void margs(Sequential& q,K x,J i) {
 S s=nullptr,nm=nullptr;
 if(xsym(x,s) || xsym(x,i,s)) {  // x is single sym, or 1st part of arg list is sym
  if(xsym(x,i+1,nm)) i++;        // increment offset if 2nd elem is sym for name
  pushback(q,s,nm,i+1,x);        // add a single module to sequential
 } else if(x->t == KS) {         // if sym vector, i.e. module type & name
  if(x->n==1 || x->n==2) {       // only 1 or 2 symbols expected
   s=kS(x)[0];                   // set module type
   if(x->n==2) nm=kS(x)[1];      // and name, if supplied
   pushback(q,s,nm,x->n,x);      // add a module from sym(s)
  } else {
   AT_ERROR("Unable to process list of ",x->n," symbols");
  }
 } else {
  TORCH_CHECK(!x->t, "Unrecognized module arg(s): ",kname(x)," supplied");
  K y=i ? kK(x)[i] : x;
  if(y->t)
   margs(q,y,0);
  else
   for(J j=0;j<y->n;++j) margs(q,kK(y)[j],0);
 }
}

KAPI seq(K x) {
 KTRY
  Sequential q,u; bool a=env().alloptions,p=xseq(x,0,q);
  if(xempty(x)) {
   return kseq(q);
  } else if(xseq(x,q) || (p && x->n==2 && xbool(x,1,a))) {
   return mtable(q,a,false);
  } else if(p && x->n==2 && xseq(x,1,u)) {
    return q->extend(*u), (K)0;
  } else if(xstate(x) || (p && x->n==2 && xstate(x,1))) {
   return pushback(q,p ? kK(x)[1] : x), p ? (K)0 : kseq(q);
  } else {
   return margs(q,x,p), p ? (K)0 : kseq(q);
  }
 KCATCH("Sequential module");
}

K mstate(K x) {
 bool a=env().alloptions; S s1=nullptr,s2=nullptr; J i; Sequential q;
 if(xseq(x,q) || (xbool(x,1,a) && x->n==2 && xseq(x,0,q))) {
  return mtable(q,a);
 } else if(xseq(x,0,q)
   && (xsym(x,1,s1) || xlong(x,1,i)) 
   && (x->n==2 || (x->n==3 && (xsym(x,2,s2) || xbool(x,2,a))))) {
  return s1 ? mchild(a,s1,s2,q) : mchild(a,i,s2,q);
 } else {
  return KERR("Unexpected arg(s) for module state");
 }
}

K seqforward(Sequential& q,K a) {
 Tensor *x,*y;
 TORCH_CHECK(!a->t && (a->n==2 || a->n==3), "forward expects 2-3 args: model/module & input tensor/arrays, e.g. (m;x) or (m;x;y)");
 x=xten(a,1);
 if(a->n==3) y=xten(a,2);
 return kten(a->n==2 ? q->forward(x ? *x : kput(a,1))
                     : q->forward(x ? *x : kput(a,1), y ? *y : kput(a,2)));
}

// ---------------------------------------------------------------------------------------
// seqattr - return requested attribute of given sequential module
// ---------------------------------------------------------------------------------------
K seqattr(const Sequential& q,Ktype k,Attr a) {
 switch(a) {
  case Attr::ptr:     return kj((intptr_t)q.get());
  case Attr::ref:     return kj(q.ptr().use_count());
  default: AT_ERROR(mapattr(a),": not implemented for sequential module");
 }
}

// ----------------------------------
// module fns defined in k namespace
// ----------------------------------
void nnfn(K x) {
 fn(x, "seq",        KFN(seq), 1);            // api function for module create/query
 fn(x, "adaptavg1d", KFN(adaptavg1d),  1);    // functional form of modules/activations
 fn(x, "adaptavg2d", KFN(adaptavg2d),  1);
 fn(x, "adaptavg3d", KFN(adaptavg3d),  1);
 fn(x, "adaptmax1d", KFN(adaptmax1d),  1);
 fn(x, "adaptmax2d", KFN(adaptmax2d),  1);
 fn(x, "adaptmax3d", KFN(adaptmax3d),  1);
 fn(x, "fmaxpool2d", KFN(fmaxpool2d),  1);
 fn(x, "fmaxpool3d", KFN(fmaxpool3d),  1);
 fn(x, "avgpool1d",  KFN(avgpool1d),   1);
 fn(x, "avgpool2d",  KFN(avgpool2d),   1);
 fn(x, "avgpool3d",  KFN(avgpool3d),   1);
 fn(x, "pad",        KFN(kpad),        1);
 fn(x, "celu",       KFN(celu),        1);
 fn(x, "elu",        KFN(elu),         1);
 fn(x, "flatten",    KFN(kflatten),    1);
 fn(x, "fold",       KFN(Fold),        1);
 fn(x, "glu",        KFN(glu),         1);
 fn(x, "hardshrink", KFN(hardshrink),  1);
 fn(x, "hardtanh",   KFN(Hardtanh),    1);
 fn(x, "leakyrelu",  KFN(leakyrelu),   1);
 fn(x, "linear",     KFN(Linear),      1);
 fn(x, "bilinear",   KFN(Bilinear),    1);
 fn(x, "logsigmoid", KFN(logsigmoid),  1);
 fn(x, "logsoftmax", KFN(logsoftmax),  1);
 fn(x, "lppool1d",   KFN(lppool1d),    1);
 fn(x, "lppool2d",   KFN(lppool2d),    1);
 fn(x, "maxpool1d",  KFN(maxpool1d),   1);
 fn(x, "maxpool2d",  KFN(maxpool2d),   1);
 fn(x, "maxpool3d",  KFN(maxpool3d),   1);
 fn(x, "normalize",  KFN(Normalize),   1);
 fn(x, "prelu",      KFN(Prelu),       1);
 fn(x, "gelu",       KFN(gelu),        1);
 fn(x, "relu",       KFN(relu),        1);
 fn(x, "relu6",      KFN(relu6),       1);
 fn(x, "rrelu",      KFN(Rrelu),       1);
 fn(x, "selu",       KFN(selu),        1);
 fn(x, "softmax",    KFN(softmax),     1);
 fn(x, "softmin",    KFN(softmin),     1);
 fn(x, "softplus",   KFN(Softplus),    1);
 fn(x, "softsign",   KFN(softsign),    1);
 fn(x, "softshrink", KFN(softshrink),  1);
 fn(x, "tanhshrink", KFN(tanhshrink),  1);
 fn(x, "threshold",  KFN(Threshold),   1);
 fn(x, "unfold",     KFN(Unfold),      1);
 fn(x, "pairwise",   KFN(Pairwise),    1);
 fn(x, "pdist",      KFN(pdist),       1);
 fn(x, "similar",    KFN(Similar),     1);
}

/*
normalize -- functional form implemented, add module?
pairwise distance & cosine similarity: in both module & functional form but forward method needs 2 input tensors
fractional pool -- try with indices registered as buffer?
embeddingbag -- forward w'defaults should work with sequential
multi-head attention -- not in 1.4, wait for patch or 1.5
*/
