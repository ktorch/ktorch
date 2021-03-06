#include "ktorch.h"
namespace nn=torch::nn;
namespace fnn=torch::nn::functional;

// ---------------------------------------------------------------------------
// mname_ - given module reference, return access to private, optional name
// mname  - given module reference return optional name
//        - also, given layer variant/layer ptr, return name or null ptr
// mlabel - demangle and simplify module type for use in error messages
// ---------------------------------------------------------------------------
const
c10::optional<std::string>& mname_(const Module& m) {return access_private::name_(m);}
c10::optional<std::string>& mname_(      Module& m) {return access_private::name_(m);}
S mname(const Module& m) {auto& s=access_private::name_(m); return const_cast<char*>(s ? (*s).c_str() : nullptr);}

std::string mlabel(const Module& x) {
 auto s=c10::demangle(typeid(x).name());
 if(!s.find("struct "))     s.erase(s.begin(),s.begin()+7);
 if(!s.find("class "))      s.erase(s.begin(),s.begin()+6);
 if(!s.find("torch::nn::")) s.erase(s.begin(),s.begin()+11);
 if(s.find("Impl",s.size()-4)==s.size()-4) s.erase(s.size()-4,s.size());
 return s;
}

std::string mlabel(const Moduleptr& x) {return mlabel(*x);}
std::string mlabel(Kmodule* x) {return mlabel(x->m);}

// ----------------------------------------------------------------------------------
// OPTION - macro to append a module option to a k dictionary given dict,name & value
// argstart - return offset in k list to begin processing module args
// anymodule - forward declare function to create a module from k args, offset & cast
// mopt - forward declare function to return module settings as k dictionary
// ----------------------------------------------------------------------------------
#define OPTION(x,k,v) dictadd(x, mset(Setting::k), v)
//static J argstart(K x,S s) {return xdict(x) ? -1 : (s ? 2 : 1);}
static J argstart(K x,S s) {return !x ? -1 : (xdict(x) ? 0 : (s ? 2 : 1));}
static AnyModule anymodule(K x,J i,S s);
static AnyModule anymodule(Cast c,const Moduleptr& m);
static K mopt(bool,bool,Cast,const Module&);

// -----------------------------------------------------------------------------------
// msym - map to/from sym & enum for module, e.g. `conv3d <-> Cast::conv3d
// msyms - parse module and optional name symbol from k arg(s), throw error if not found
// mset - map to/from sym & enum for module options, e.g. `bias <-> Setting::bias
// mpos - throw error if too many positional arguments
// mpair - throw error if unrecognized name in name-value pairs
// mkeys - keys for dict/table of module state: `depth`module`name`options`parms`buffers
// -----------------------------------------------------------------------------------
S msym(Cast c) {
 for(auto& m:env().module) if(c==std::get<1>(m)) return std::get<0>(m);
 TORCH_ERROR("unrecognized module: cannot translate enumeration ",(I)c," to symbol");
}

Cast msym(S s) {
 for(const auto& m:env().module) if(s==std::get<0>(m)) return std::get<1>(m);
 TORCH_ERROR("unrecognized module: ",s);
}

static void msyms(K x,S& s,S& nm) {
 nm=nullptr;
 if(x->t == -KS) {
  s=x->s;
 } else if(x->t == KS) {
  TORCH_CHECK(x->n>0, "module: empty symbol list");
  s=kS(x)[0];
  if(x->n>1) nm=kS(x)[1];
 } else if(!x->t) {
  TORCH_CHECK(x->n>0, "module: empty list");
  TORCH_CHECK(kK(x)[0]->t==-KS, "module: no symbol found, ",kstring(x));
  s=kK(x)[0]->s;
  if(x->n>1 && kK(x)[1]->t==-KS) nm=kK(x)[1]->s;
 } else {
  TORCH_ERROR("module: unrecognized arg(s), ", kstring(x));
 }
}

static S mset(Setting x) {
 for(auto& m:env().mset) if(x == std::get<1>(m)) return std::get<0>(m);
 TORCH_ERROR("unrecognized module option: ",(I)x);
}

static Setting mset(S x,Cast c=Cast::undefined);
static Setting mset(S x,Cast c) {
 for(const auto& m:env().mset) if(x == std::get<0>(m)) return std::get<1>(m);
 if(c == Cast::undefined)
  TORCH_ERROR("unrecognized option: `",x);
 else
  TORCH_ERROR(msym(c),": unrecognized option `",x);
}

static void mpos(K x,Cast c,J n) {
 TORCH_ERROR(msym(c),": expecting up to ",n," positional args, ",xlen(x)," given");
}

static void mpair(Cast c,const Pairs& p) {
 TORCH_ERROR(msym(c)," option: ",p.k," not recognized");
}

static K mkeys(bool b) {
 K x=ktn(KS, b ? 6 : 4);
 kS(x)[0]=statekey(State::depth);
 kS(x)[1]=statekey(State::module);
 kS(x)[2]=statekey(State::name);
 kS(x)[3]=statekey(State::options);
 if(b) {
  kS(x)[4]=statekey(State::parms);
  kS(x)[5]=statekey(State::buffers);
 }
 return x;
}

// -----------------------------------------------------------------------
// mcast - given generic module, return api enumeration, e.g. Cast::linear
// msym  - given generic module, return api symbol, e.g. `linear
// -----------------------------------------------------------------------
static Cast mcast(size_t h) {
 for(const auto& m:env().module)
  if(std::get<2>(m)==h) return std::get<1>(m);
 return Cast::undefined;
}

Cast mcast(const Module& m) {return mcast(typeid(m).hash_code());}

static S msym(size_t h) {
 for(const auto& m:env().module)
  if(std::get<2>(m)==h) return std::get<0>(m);
 return nullsym();
}

S msym(const Module& m) {return msym(typeid(m).hash_code());}

// -----------------------------------------------------------------------------------
// seqlist - enlist x, only allow symbol scalar
// seq - convenience function to enlist all but 1st arg to build sequential arg list
// -----------------------------------------------------------------------------------
static K seqlist(K x) {
 K r;
 if(x->t<0) {
  TORCH_CHECK(x->t == -KS, "scalar expected to be a symbol, given a ",kname(x));
  r=ktn(KS,1), kS(r)[0]=x->s;
 } else {
  r=knk(1,r1(x));
 }
 return r;
}

KAPI seq(K x) {
 KTRY
  K r;
  if(x->t<0) {
   TORCH_CHECK(x->t==-KS, "seq: expecting module symbol, given ",kname(x),", ",kstring(x));
   r=r1(x);
  } else if(x->t>0) {
   TORCH_CHECK(x->t==KS, "seq: expecting module symbols, given ",kname(x),", ",kstring(x));
   TORCH_CHECK(x->n>0,   "seq: expecting at least one module symbol, given  empty list");
   r=ktn(0,x->n); kK(r)[0]=ks(kS(x)[0]);
   for(J i=1;i<x->n;++i) {
    kK(r)[i]=ktn(KS,1); kS(kK(r)[i])[0]=kS(x)[i];
   }
  } else {
   TORCH_CHECK(x->n>0, "seq: empty list");
   r=ktn(0,x->n);
   kK(r)[0]=r1(kK(x)[0]);
   for(J i=1;i<x->n;++i)
    kK(r)[i]=seqlist(kK(x)[i]);
  }
  return r;
 KCATCH("seq");
}

// ------------------------------------------------------------------------------
// kmodule - allocate object to store a module pointer (class defaults to module) 
// to - given module & options, change device/data type
// ------------------------------------------------------------------------------
K kmodule(Cast c,const Moduleptr& m,Class a) {return kptr(new Kmodule(a,c,m));}

void to(Module& m,const TensorOptions& o,bool a) {
 TORCH_CHECK( !(o.has_layout() || o.has_requires_grad() || o.has_pinned_memory() || o.has_memory_format()),
             "to: converts device & type, but cannot be used for layout,gradient,pinned memory or memory format");
 auto s=torch::typeMetaToScalarType(o.dtype());
 if(o.has_device() && o.has_dtype()) m.to(o.device(),s,a);
 else if(o.has_device())             m.to(o.device(),a);
 else                                m.to(s,a);
}

K to(Kmodule* m,const TensorOptions& o,bool a) {to(*m->m,o,a); return(K)0;}

// --------------------------------------------------------------------------------------
// container - given module/module cast, return true if container module
// parmdict - parameter dictionary handles "options" of dictionary of tensors or k arrays
// --------------------------------------------------------------------------------------
static bool container(Cast c) {
 switch(c) {
  case Cast::sequential:
  case Cast::seqnest:
  case Cast::seqjoin:
  case Cast::moduledict:
  case Cast::modulelist:
  case Cast::parmdict:
  case Cast::fork:
  case Cast::nbeats:
  case Cast::recur:
  case Cast::residual:
  case Cast::transform:
  case Cast::base:
   return true;
  default: return false;
 }
}

static bool container(const Module& m) {
 if     (m.as<nn::Sequential>())    return true;
 else if(m.as<SeqNest>())           return true;
 else if(m.as<SeqJoin>())           return true;
 else if(m.as<nn::ModuleDict>())    return true;
 else if(m.as<nn::ModuleList>())    return true;
 else if(m.as<nn::ParameterDict>()) return true;
 else if(m.as<Fork>())              return true;
 else if(m.as<NBeats>())            return true;
 else if(m.as<Recur>())             return true;
 else if(m.as<Residual>())          return true;
 else if(m.as<Transform>())         return true;
 else if(m.as<BaseModule>())        return true;
 else                               return false;
}

static bool container(const Moduleptr& p) {return p ? container(*p) : false;}

static Moduleptr parmdict(K x,J i) {
 if(!x || xnone(x,i))
  return nn::ParameterDict().ptr();
 else if(auto *d=xtensordict(x,i))
  return nn::ParameterDict(*d).ptr();
 else if(xdict(x) || xdict(x,i))
  return nn::ParameterDict(kputd(xdict(x) ? x : kK(x)[i])).ptr();
 else
  TORCH_ERROR("module: parameter dictionary expects a k dictionary or an allocated dictionary of tensors, given ",kname(x,i));
}

// -------------------------------------------------------------------------------------------------
// mstack - adjust stack given depth,then populate a stack of all intermediate container modules
// mfirst - return first module put on stack (pare down stack, signal error if given empty stack)
// mresult - if existing module, update result type & return null, else return new module structure
// -------------------------------------------------------------------------------------------------
static void mstack(size_t d,const Moduleptr& m,Modules& q) {
 while(q.size()>d) q.pop();
 if(container(m)) {
  q.push(m);
  for(const auto& i:m->children())
   mstack(d+1,i,q);
 }
}

static Modules mstack(Kmodule *m) {
 Modules q;
 if(m) {
  if(container(m->m))
   mstack(0,m->m,q);
  else
   q.push(m->m);
 }
 return q;
}

static Moduleptr mfirst(Modules& q) {
 TORCH_CHECK(q.size(), "empty module stack -- cannot get originating module");
 while(q.size()>1) q.pop();
 return q.top();
}

static K mresult(Kmodule *m,Cast c,Modules& q) {
 const auto& a=mfirst(q);
 return m ? (K)0 : kmodule(c,a);
}

// ------------------------------------------------------------------------------------
// mforward - given layer, run forward calc on tensor x and optional y,z tensors
// ------------------------------------------------------------------------------------
Output mforward(Cast c,Module& m,const Tensor& x) {
 switch(c) {
  case Cast::adaptavg1d:      return m.as<nn::AdaptiveAvgPool1d>()->forward(x);
  case Cast::adaptavg2d:      return m.as<nn::AdaptiveAvgPool2d>()->forward(x);
  case Cast::adaptavg3d:      return m.as<nn::AdaptiveAvgPool3d>()->forward(x);
  case Cast::adaptmax1d:      return m.as<nn::AdaptiveMaxPool1d>()->forward(x);
  case Cast::adaptmax2d:      return m.as<nn::AdaptiveMaxPool2d>()->forward(x);
  case Cast::adaptmax3d:      return m.as<nn::AdaptiveMaxPool3d>()->forward(x);
  case Cast::adrop:           return m.as<nn::AlphaDropout>()->forward(x);
//case Cast::attention:       return m.as<nn::MultiheadAttention>()->forward(x);
// too few arguments to function call, expected at least 3, have 1
  case Cast::avgpool1d:       return m.as<nn::AvgPool1d>()->forward(x);
  case Cast::avgpool2d:       return m.as<nn::AvgPool2d>()->forward(x);
  case Cast::avgpool3d:       return m.as<nn::AvgPool3d>()->forward(x);
  case Cast::base:            return m.as<BaseModule>()->forward(x);
  case Cast::batchnorm1d:     return m.as<nn::BatchNorm1d>()->forward(x);
  case Cast::batchnorm2d:     return m.as<nn::BatchNorm2d>()->forward(x);
  case Cast::batchnorm3d:     return m.as<nn::BatchNorm3d>()->forward(x);
  case Cast::celu:            return m.as<nn::CELU>()->forward(x);
  case Cast::conv1d:          return m.as<nn::Conv1d>()->forward(x);
  case Cast::conv2d:          return m.as<nn::Conv2d>()->forward(x);
  case Cast::conv3d:          return m.as<nn::Conv3d>()->forward(x);
  case Cast::convtranspose1d: return m.as<nn::ConvTranspose1d>()->forward(x);
  case Cast::convtranspose2d: return m.as<nn::ConvTranspose2d>()->forward(x);
  case Cast::convtranspose3d: return m.as<nn::ConvTranspose3d>()->forward(x);
  case Cast::crossmap2d:      return m.as<nn::CrossMapLRN2d>()->forward(x);
  case Cast::drop:            return m.as<nn::Dropout>()->forward(x);
  case Cast::drop2d:          return m.as<nn::Dropout2d>()->forward(x);
  case Cast::drop3d:          return m.as<nn::Dropout3d>()->forward(x);
  case Cast::elu:             return m.as<nn::ELU>()->forward(x);
  case Cast::embed:           return m.as<nn::Embedding>()->forward(x);
  case Cast::embedbag:        return m.as<nn::EmbeddingBag>()->forward(x);
  case Cast::encoder:         return m.as<nn::TransformerEncoder>()->forward(x);
  case Cast::encoderlayer:    return m.as<nn::TransformerEncoderLayer>()->forward(x);
  case Cast::expand:          return m.as<Expand>()->forward(x);
  case Cast::fadrop:          return m.as<nn::FeatureAlphaDropout>()->forward(x);
  case Cast::flatten:         return m.as<nn::Flatten>()->forward(x);
  case Cast::fmaxpool2d:      return m.as<nn::FractionalMaxPool2d>()->forward(x);
  case Cast::fmaxpool3d:      return m.as<nn::FractionalMaxPool3d>()->forward(x);
  case Cast::fold:            return m.as<nn::Fold>()->forward(x);
  case Cast::fork:            return m.as<Fork>()->forward(x);
  case Cast::gelu:            return m.as<nn::GELU>()->forward(x);
  case Cast::glu:             return m.as<nn::GLU>()->forward(x);
  case Cast::groupnorm:       return m.as<nn::GroupNorm>()->forward(x);
  case Cast::gru:             return m.as<nn::GRU>()->forward(x);
  case Cast::hardshrink:      return m.as<nn::Hardshrink>()->forward(x);
  case Cast::hardtanh:        return m.as<nn::Hardtanh>()->forward(x);
  case Cast::identity:        return m.as<nn::Identity>()->forward(x);
  case Cast::indexselect:     return m.as<IndexSelect>()->forward(x);
  case Cast::instancenorm1d:  return m.as<nn::InstanceNorm1d>()->forward(x);
  case Cast::instancenorm2d:  return m.as<nn::InstanceNorm2d>()->forward(x);
  case Cast::instancenorm3d:  return m.as<nn::InstanceNorm3d>()->forward(x);
//case Cast::interpolate:     return m.as<nn::interpolate>()->forward(x);
  case Cast::layernorm:       return m.as<nn::LayerNorm>()->forward(x);
  case Cast::leakyrelu:       return m.as<nn::LeakyReLU>()->forward(x);
  case Cast::linear:          return m.as<nn::Linear>()->forward(x);
  case Cast::localnorm:       return m.as<nn::LocalResponseNorm>()->forward(x);
  case Cast::logsigmoid:      return m.as<nn::LogSigmoid>()->forward(x);
  case Cast::logsoftmax:      return m.as<nn::LogSoftmax>()->forward(x);
  case Cast::lppool1d:        return m.as<nn::LPPool1d>()->forward(x);
  case Cast::lppool2d:        return m.as<nn::LPPool2d>()->forward(x);
  case Cast::lstm:            return m.as<LSTM>()->forward(x);
  case Cast::maxpool1d:       return m.as<nn::MaxPool1d>()->forward(x);
  case Cast::maxpool2d:       return m.as<nn::MaxPool2d>()->forward(x);
  case Cast::maxpool3d:       return m.as<nn::MaxPool3d>()->forward(x);
  case Cast::mish:            return m.as<nn::Mish>()->forward(x);
//case Cast::normalize:       return m.as<nn::normalize>()->forward(x);
  case Cast::nbeats:          return m.as<NBeats>()->forward(x);
  case Cast::onehot:          return m.as<OneHot>()->forward(x);
  case Cast::pad:             return m.as<Pad>()->forward(x);
  case Cast::pad1d:           return m.as<nn::ConstantPad1d>()->forward(x);
  case Cast::pad2d:           return m.as<nn::ConstantPad2d>()->forward(x);
  case Cast::pad3d:           return m.as<nn::ConstantPad3d>()->forward(x);
  case Cast::prelu:           return m.as<nn::PReLU>()->forward(x);
  case Cast::randomcrop:      return m.as<RandomCrop>()->forward(x);
  case Cast::randomflip:      return m.as<RandomFlip>()->forward(x);
  case Cast::recur:           return m.as<Recur>()->forward(x);
  case Cast::reflect1d:       return m.as<nn::ReflectionPad1d>()->forward(x);
  case Cast::reflect2d:       return m.as<nn::ReflectionPad2d>()->forward(x);
  case Cast::relu:            return m.as<nn::ReLU>()->forward(x);
  case Cast::relu6:           return m.as<nn::ReLU6>()->forward(x);
  case Cast::replicate1d:     return m.as<nn::ReplicationPad1d>()->forward(x);
  case Cast::replicate2d:     return m.as<nn::ReplicationPad2d>()->forward(x);
  case Cast::replicate3d:     return m.as<nn::ReplicationPad3d>()->forward(x);
  case Cast::residual:        return m.as<Residual>()->forward(x);
  case Cast::reshape:         return m.as<Reshape>()->forward(x);
  case Cast::rnn:             return m.as<nn::RNN>()->forward(x);
  case Cast::rrelu:           return m.as<nn::RReLU>()->forward(x);
  case Cast::select:          return m.as<Select>()->forward(x);
  case Cast::selu:            return m.as<nn::SELU>()->forward(x);
  case Cast::seqnest:         return m.as<SeqNest>()->forward(x);
  case Cast::sequential:      return m.as<nn::Sequential>()->forward(x);
  case Cast::sigmoid:         return m.as<nn::Sigmoid>()->forward(x);
  case Cast::silu:            return m.as<nn::SiLU>()->forward(x);
  case Cast::softmax:         return m.as<nn::Softmax>()->forward(x);
  case Cast::softmax2d:       return m.as<nn::Softmax2d>()->forward(x);
  case Cast::softmin:         return m.as<nn::Softmin>()->forward(x);
  case Cast::softplus:        return m.as<nn::Softplus>()->forward(x);
  case Cast::softshrink:      return m.as<nn::Softshrink>()->forward(x);
  case Cast::softsign:        return m.as<nn::Softsign>()->forward(x);
  case Cast::squeeze:         return m.as<Squeeze>()->forward(x);
  case Cast::tanh:            return m.as<nn::Tanh>()->forward(x);
  case Cast::tanhshrink:      return m.as<nn::Tanhshrink>()->forward(x);
  case Cast::threshold:       return m.as<nn::Threshold>()->forward(x);
  case Cast::transform:       return m.as<Transform>()->forward(x);
  case Cast::unfold:          return m.as<nn::Unfold>()->forward(x);
  case Cast::unsqueeze:       return m.as<Unsqueeze>()->forward(x);
  case Cast::upsample:        return m.as<nn::Upsample>()->forward(x);
  case Cast::zeropad2d:       return m.as<nn::ZeroPad2d>()->forward(x);
  case Cast::zscore:          return m.as<Zscore>()->forward(x);
  default: TORCH_ERROR("forward calculation with single tensor argument not implemented for ",msym(c)," module");
 }
}

Output mforward(Cast c,Module& m,const Tensor& x,const Tensor& y) {
 switch(c) {
  case Cast::bilinear:        return m.as<nn::Bilinear>()->forward(x,y);
  case Cast::cat:             return m.as<Cat>()->forward(x,y);
  case Cast::decoder:         return m.as<nn::TransformerDecoder>()->forward(x,y);
  case Cast::decoderlayer:    return m.as<nn::TransformerDecoderLayer>()->forward(x,y);
  case Cast::encoder:         return m.as<nn::TransformerEncoder>()->forward(x,y);
  case Cast::encoderlayer:    return m.as<nn::TransformerEncoderLayer>()->forward(x,y);
  case Cast::gru:             return m.as<nn::GRU>()->forward(x,y);
  case Cast::mul:             return m.as<Mul>()->forward(x,y);
  case Cast::pairwise:        return m.as<nn::PairwiseDistance>()->forward(x,y);
  case Cast::recur:           return m.as<Recur>()->forward(x,y);
  case Cast::rnn:             return m.as<nn::RNN>()->forward(x,y);
  case Cast::seqjoin:         return m.as<SeqJoin>()->forward(x,y);
  case Cast::sequential:      return m.as<nn::Sequential>()->forward(x,y);
  case Cast::similar:         return m.as<nn::CosineSimilarity>()->forward(x,y);
  case Cast::transformer:     return m.as<nn::Transformer>()->forward(x,y);
  default: TORCH_ERROR("forward calculation with 2 tensor inputs not implemented for ",msym(c)," module");
 }
}

Output mforward(Cast c,Module& m,const Tensor& x,const Tensor& y,const Tensor& z) {
 switch(c) {
  case Cast::decoder:         return m.as<nn::TransformerDecoder>()->forward(x,y,z);
  case Cast::decoderlayer:    return m.as<nn::TransformerDecoderLayer>()->forward(x,y,z);
  case Cast::encoder:         return m.as<nn::TransformerEncoder>()->forward(x,y,z);
  case Cast::encoderlayer:    return m.as<nn::TransformerEncoderLayer>()->forward(x,y,z);
  case Cast::lstm:            return m.as<LSTM>()->forward(x,y,z);
  case Cast::transformer:     return m.as<nn::Transformer>()->forward(x,y,z);
  case Cast::recur:           return m.as<Recur>()->forward(x,y,z);
  case Cast::sequential:      return m.as<nn::Sequential>()->forward(x,y,z);
  default: TORCH_ERROR("forward calculation with 3 tensor inputs not implemented for ",msym(c)," module");
 }
}

// ----------------------------------------------------------------------------------------------------
// covers of input checking fns with error msg specific to module settings and module names:
// ----------------------------------------------------------------------------------------------------
// mbool - check args for boolean, else error w'module & option name
// code  - check args for symbol,  else error w'module & option name
// otype - check args for optional data type (null symbol -> nullopt)
// int64 - check args for long int, else error w'module & option
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

static S code(K x,J i,Cast c,Setting s) {
 S m;
 TORCH_CHECK(xsym(x,i,m), msym(c)," ",mset(s),": expected symbol, given ",kname(x,i));
 return m;
}

static S code(const Pairs& p,Cast c) {
 TORCH_CHECK(p.t==-KS, msym(c)," ",p.k,": expected symbol, given ",kname(p.t));
 return p.s;
}

static c10::optional<Dtype> otype(S s) {if(nullsym(s)) return c10::nullopt; else return stype(s);}
static c10::optional<Dtype> otype(K x,J i,Cast c,Setting s) {return otype(code(x,i,c,s));}
static c10::optional<Dtype> otype(const Pairs& p,Cast c)    {return otype(code(p,c));}

static int64_t int64(K x,J i,Cast c,Setting s) {
 int64_t n;
 TORCH_CHECK(xint64(x,i,n), msym(c)," ",mset(s),": expected long scalar, given ",kname(x,i));
 return n;
}

static int64_t int64(const Pairs& p,Cast c) {
 TORCH_CHECK(p.t==-KJ, msym(c)," ",p.k,": expected long scalar, given ",kname(p.t));
 return p.j;
}

static c10::optional<int64_t> int64n(K x,J i,Cast c,Setting s) {auto n=int64(x,i,c,s); if(null(n)) return c10::nullopt; else return n;}
static c10::optional<int64_t> int64n(const Pairs& p,Cast c)    {auto n=int64(p,c);     if(null(n)) return c10::nullopt; else return n;}

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

// ----------------------------------------------------------------------------------------
// mlongs - check for long(s), return vector else error specific to module and setting
// ltensor - define tensor from long(s), else error specific to module & setting
// ftensor - define tensor from long/float/double(s), else error specific to setting
// mdoubles - check for double(s), return vector else error specific to module and setting
// ----------------------------------------------------------------------------------------
static LongVector mlongs(K x,J i,Cast c,Setting s) {
 IntArrayRef a;
 TORCH_CHECK(xsize(x,i,a), msym(c)," ",mset(s),": expected long(s), given ",kname(x,i));
 return a.vec();
}

static LongVector mlongs(const Pairs& p,Cast c) {
 IntArrayRef a;
 TORCH_CHECK(p.t==-KJ || p.t==KJ, msym(c)," ",p.k,": expected long(s), given ",kname(p.t));
 psize(p,a);
 return a.vec();
}

static Tensor ltensor(K x,J i,Cast c,Setting s) {
 Tensor t; if(!xten(x,i,t)) t=kput(x,i);
 TORCH_CHECK(t.dtype()==torch::kLong, msym(c)," ",mset(s),": long(s) expected, given ",t.dtype(),"(s)");
 return t;
}

static Tensor ltensor(const Pairs& p,Cast c) {
 Tensor t; pten(p,t);
 TORCH_CHECK(t.dtype()==torch::kLong, msym(c)," ",p.k,": long(s) expected, given ",t.dtype(),"(s)");
 return t;
}

static Tensor ftensor(K x,J i,Cast c,Setting s) {
 Tensor t; if(!xten(x,i,t)) t=kput(x,i); if(t.dtype()==torch::kLong) t=t.to(torch::kDouble);
 TORCH_CHECK(t.is_floating_point(), msym(c)," ",mset(s),": double(s) expected, given ",t.dtype(),"(s)");
 return t;
}

static Tensor ftensor(const Pairs& p,Cast c) {
 Tensor t; pten(p,t); if(t.dtype()==torch::kLong) t=t.to(torch::kDouble);
 TORCH_CHECK(t.is_floating_point(), msym(c)," ",p.k,": double(s) expected, given ",t.dtype(),"(s)");
 return t;
}

static DoubleVector mdoubles(K x,J i,Cast c,Setting s) {
 J n; F *f; IntArrayRef a; DoubleVector v;
 if(xsize(x,i,a)) {
  for(const auto j:a) v.push_back(j);
 } else if(xdouble(x,i,n,f)) {
  v=DoubleArrayRef(f,n).vec();
 } else {
  TORCH_ERROR(msym(c)," ",mset(s),": expected double(s), given ",kname(x,i));
 }
 return v;
}

static DoubleVector mdoubles(const Pairs& p,Cast c) {
 DoubleVector v;
 if(p.t==-KJ || p.t==KJ) {
  IntArrayRef a; psize(p,a);
  for(const auto j:a) v.push_back(j);
 } else if(p.t==-KF || p.t==KF) {
  DoubleArrayRef a; pdoubles(p,a); v=a.vec();
 } else {
  TORCH_ERROR(msym(c)," ",p.k,": expected double(s), given ",kname(p.t));
 }
 return v;
}


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
 return null(j) ? Exoptional<D>(c10::nullopt) : Exoptional<D>(j);
}

template<size_t D> static Exoptional<D> exoptional(K x) {
 auto a=Exoptional<D>(IntArrayRef((int64_t*)kJ(x),x->n));
 for(J i=0;i<x->n;++i) if(null((*a)[i].value())) (*a)[i]=c10::nullopt;
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
    default: mpos(x,c,i+j); break;
  }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::in:       o.num_features(int64(p,c)); in=true; break;
   case Setting::eps:      o.eps(mdouble(p,c)); break;
   case Setting::momentum: o.momentum(mdouble(p,c)); break;
   case Setting::affine:   o.affine(mbool(p,c)); break;
   case Setting::track:    o.track_running_stats(mbool(p,c)); break;
   default: mpair(c,p); break;
  }
 TORCH_CHECK(in,msym(c),": number of input features not defined");
 return o;
}

template<typename O> static K batchnorm(bool a,const O& o) {
 K x=KDICT; O d(o.num_features());
 OPTION(x, in, kj(o.num_features()));
 if(a || (o.eps()      != d.eps()))      OPTION(x, eps,       kf(o.eps()));
 if(a || (o.momentum() != d.momentum())) OPTION(x, momentum,  kf(momentum(o.momentum())));
 if(a || (o.affine()   != d.affine()))   OPTION(x, affine,    kb(o.affine()));
 if(a || (o.track_running_stats() != d.track_running_stats())) OPTION(x, track, kb(o.track_running_stats()));
 return x;
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
    default: mpos(x,c,i+j); break;
  }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::size:  o.size(int64(p,c)); sz=true; break;
   case Setting::alpha: o.alpha(mdouble(p,c)); break;
   case Setting::beta:  o.beta(mdouble(p,c)); break;
   case Setting::k:     b ? o.k(mdouble(p,c)) : o.k(int64(p,c)); break;
   default: mpair(c,p); break;
  }
 TORCH_CHECK(sz, msym(c),": specify no. of neighboring channels to use for normalization");
 return o;
}

template<typename O> static K localnorm(bool a,Cast c,const O& o) {
 K x=KDICT; O d(o.size());
 OPTION(x, size, kj(o.size()));
 if(a || (o.alpha() != d.alpha())) OPTION(x, alpha, kf(o.alpha()));
 if(a || (o.beta()  != d.beta()))  OPTION(x, beta,  kf(o.beta()));
 if(a || (o.k()     != d.k()))     OPTION(x, k,     c==Cast::localnorm ? kf(o.k()) : kj(o.k()));
 return x;
}

// --------------------------------------------------------------------------------------
// groupnorm - group norm, get/set number of groups,channels,eps,affine flag
// --------------------------------------------------------------------------------------
static nn::GroupNormOptions groupnorm(K x,J i,Cast c) {
 nn::GroupNormOptions o(0,0);
 bool g=false,h=false; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
   switch(j) {
    case 0: o.num_groups(int64(x,i+j,c,Setting::groups)); g=true; break;
    case 1: o.num_channels(int64(x,i+j,c,Setting::channels)); h=true; break;
    case 2: o.eps(mdouble(x,i+j,c,Setting::eps)); break;
    case 3: o.affine(mbool(x,i+j,c,Setting::affine)); break;
    default: mpos(x,c,i+j); break;
  }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::groups:   o.num_groups(int64(p,c)); g=true; break;
   case Setting::channels: o.num_channels(int64(p,c)); h=true; break;
   case Setting::eps:      o.eps(mdouble(p,c)); break;
   case Setting::affine:   o.affine(mbool(p,c)); break;
   default: mpair(c,p); break;
  }
 TORCH_CHECK(g, msym(c),": specify no. of groups to separate the channels into");
 TORCH_CHECK(h, msym(c),": specify no. of channels expected in input");
 return o;
}

static K groupnorm(bool a,const nn::GroupNormOptions& o) {
 K x=KDICT; nn::GroupNormOptions d(o.num_groups(),o.num_channels());
 OPTION(x, groups,   kj(o.num_groups()));
 OPTION(x, channels, kj(o.num_channels()));
 if(a || (o.eps()    != d.eps()))    OPTION(x, eps,    kf(o.eps()));
 if(a || (o.affine() != d.affine())) OPTION(x, affine, kb(o.affine()));
 return x;
}

// --------------------------------------------------------------------------------------
// layernorm - get/set shape,eps,affine flag for layer normalization
// --------------------------------------------------------------------------------------
static nn::LayerNormOptions layernorm(K x,J i,Cast c) {
 nn::LayerNormOptions o({}); Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
   switch(j) {
    case 0: o.normalized_shape(mlongs(x,i+j,c,Setting::shape)); break;
    case 1: o.eps(mdouble(x,i+j,c,Setting::eps)); break;
    case 2: o.elementwise_affine(mbool(x,i+j,c,Setting::affine)); break;
    default: mpos(x,c,i+j); break;
  }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::shape:  o.normalized_shape(mlongs(p,c)); break;
   case Setting::eps:    o.eps(mdouble(p,c)); break;
   case Setting::affine: o.elementwise_affine(mbool(p,c)); break;
   default: mpair(c,p); break;
  }
 TORCH_CHECK(o.normalized_shape().size(), msym(c),": no normalized shape given");
 return o;
}

static K layernorm(bool a,const nn::LayerNormOptions& o) {
 K x=KDICT; nn::LayerNormOptions d(o.normalized_shape());
 OPTION(x, shape, klist(o.normalized_shape().size(),o.normalized_shape().data()));
 if(a || (o.eps()    != d.eps())) OPTION(x, eps, kf(o.eps()));
 if(a || (o.elementwise_affine() != d.elementwise_affine())) OPTION(x, affine, kb(o.elementwise_affine()));
 return x;
}

// --------------------------------------------------------------------------------------
// normalize - pytorch has functional form only, no module as of version 1.7
// --------------------------------------------------------------------------------------
static fnn::NormalizeFuncOptions normalize(K x,J i,Cast c,Tensor& r) {
 Pairs p; J n=xargc(x,i,p); fnn::NormalizeFuncOptions o;
 if(n>0 && xten(x,i+n-1,r)) n--;
 for(J j=0;j<n;++j)
  switch(j) {
   case 0: o.p(mdouble(x,i+j,c,Setting::p)); break;
   case 1: o.dim(int64(x,i+j,c,Setting::dim)); break;
   case 2: o.eps(mdouble(x,i+j,c,Setting::eps)); break;
   default: mpos(x,c,i+j); break;
  }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::p: o.p(mdouble(p,c)); break;
   case Setting::dim: o.dim(int64(p,c)); break;
   case Setting::eps: o.eps(mdouble(p,c)); break;
   case Setting::out: if(!pempty(p)) pten(p,r);
   default: mpair(c,p); break;
  }
 if(r.defined()) 
  o.out(r);
 return o;
}

static K normalize(bool a,const fnn::NormalizeFuncOptions& o) {
 K x=KDICT; const fnn::NormalizeFuncOptions d;
 if(a || o.p()   != d.p())   OPTION(x, p, kf(o.p()));
 if(a || o.dim() != d.dim()) OPTION(x, dim, kj(o.dim()));
 if(a || o.eps() != d.eps()) OPTION(x, eps, kj(o.eps()));
 return x;
}

KAPI Normalize(K x) {
 KTRY
  namespace f=fnn;
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
// padmode - translate symbol to variant used for padding mode
// padsym - translate symbol to padding for same or valid
// convpad - translate input(symbol or long(s)) into padding for convolution
// conv - create 1-3d convolution, set dictionary given module
//        with version 1.4, the c++ ConvImpl class was split into regular & transposed
//        ConvOptions & ConvTransOptions have different members, 
// convtran - similar to conv() except adds output_padding and changes position order
// --------------------------------------------------------------------------------------
static nn::detail::conv_padding_mode_t padmode(S s) {
 switch(emap(s)) {
  case Enum::zeros:     return torch::kZeros;
  case Enum::reflect:   return torch::kReflect;
  case Enum::replicate: return torch::kReplicate;
  case Enum::circular:  return torch::kCircular;
  default: TORCH_ERROR("unrecognized padding mode: ",s); break;
 }
}

template<size_t D> static nn::detail::conv_padding_t<D> padsym(S s,Cast c) {
 switch(emap(s)) {
  case Enum::same:  return torch::kSame;
  case Enum::valid: return torch::kValid;
  default: TORCH_ERROR(msym(c),": unrecognized padding: ",s); break;
 }
}

template<size_t D> static nn::detail::conv_padding_t<D> convpad(K x,J i,Cast c) {
 S s; return xsym(x,i,s) ? padsym<D>(s,c) : exarray<D>(x,i,c,Setting::pad);
}

template<size_t D> static nn::detail::conv_padding_t<D> convpad(const Pairs& p,Cast c) {
 return p.t == -KS ? padsym<D>(p.s,c) : exarray<D>(p,c);
}

template<size_t D> static nn::ConvOptions<D> conv(K x,J i,Cast c) {
 nn::ConvOptions<D> o(0,0,0);
 bool in=false,out=false,sz=false; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
   switch(j) {
    case 0: o.in_channels (int64(x,i+j,c,Setting::in));        in=true; break;
    case 1: o.out_channels(int64(x,i+j,c,Setting::in));       out=true; break;
    case 2: o.kernel_size (exarray<D>(x,i+j,c,Setting::size)); sz=true; break;
    case 3: o.stride      (exarray<D>(x,i+j,c,Setting::stride));   break;
    case 4: o.padding     (convpad<D>(x,i+j,c));                   break;
    case 5: o.dilation    (exarray<D>(x,i+j,c,Setting::dilate));   break;
    case 6: o.groups      (int64(x,i+j,c,Setting::groups));        break;
    case 7: o.bias        (mbool    (x,i+j,c,Setting::bias));      break;
    case 8: o.padding_mode(padmode(code(x,i+j,c,Setting::padmode))); break;
    default: mpos(x,c,i+j); break;
  }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::in:        o.in_channels (int64(p,c));     in=true; break;
   case Setting::out:       o.out_channels(int64(p,c));    out=true; break;
   case Setting::size:      o.kernel_size (exarray<D>(p,c)); sz=true; break;
   case Setting::stride:    o.stride      (exarray<D>(p,c)); break;
   case Setting::pad:       o.padding     (convpad<D>(p,c)); break;
   case Setting::dilate:    o.dilation    (exarray<D>(p,c)); break;
   case Setting::groups:    o.groups      (int64(p,c));     break;
   case Setting::bias:      o.bias        (mbool(p,c));     break;
   case Setting::padmode:   o.padding_mode(padmode(code(p,c)));   break;
   default: mpair(c,p); break;
  }
 TORCH_CHECK(in,  msym(c),": number of input channels not defined");
 TORCH_CHECK(out, msym(c),": number of output channels not defined");
 TORCH_CHECK(sz,  msym(c),": no kernel size(s) given");
 return o;
}

template<size_t D> static nn::ConvTransposeOptions<D> convtran(K x,J i,Cast c) {
 nn::ConvTransposeOptions<D> o(0,0,0);
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
    case 9: o.padding_mode  (padmode(code(x,i+j,c,Setting::padmode))); break;
    default: mpos(x,c,i+j); break;
  }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::in:        o.in_channels   (int64(p,c));      in=true; break;
   case Setting::out:       o.out_channels  (int64(p,c));     out=true; break;
   case Setting::size:      o.kernel_size   (exarray<D>(p,c)); sz=true; break;
   case Setting::stride:    o.stride        (exarray<D>(p,c)); break;
   case Setting::pad:       o.padding       (exarray<D>(p,c)); break;
   case Setting::outpad:    o.output_padding(exarray<D>(p,c)); break;
   case Setting::groups:    o.groups        (int64(p,c));      break;
   case Setting::bias:      o.bias          (mbool(p,c));      break;
   case Setting::dilate:    o.dilation      (exarray<D>(p,c)); break;
   case Setting::padmode:   o.padding_mode(padmode(code(p,c)));break;
   default: mpair(c,p); break;
  }
 TORCH_CHECK(in,  msym(c), ": number of input channels not defined");
 TORCH_CHECK(out, msym(c), ": number of output channels not defined");
 TORCH_CHECK(sz,  msym(c), ": no kernel size(s) given");
 return o;
}

template<size_t D>static void convpad(K x,bool a,const nn::detail::conv_padding_t<D>& d,const nn::detail::conv_padding_t<D>& o) {
 if(auto p=c10::get_if<ExpandingArray<D>>(&o)) {
  auto pd=c10::get_if<ExpandingArray<D>>(&d);
  if(a || !pd || **pd != **p)
   OPTION(x, pad, KEX((*p)));
 } else if(a || d.index() != o.index()) {
  if(c10::get_if<torch::enumtype::kSame>(&o)) {
   OPTION(x, pad, ks(emap(Enum::same)));
  } else if(c10::get_if<torch::enumtype::kValid>(&o)) {
   OPTION(x, pad, ks(emap(Enum::valid)));
  } else {
   TORCH_ERROR("unrecognized convolution padding");
  }
 }
}

template<size_t D> static K conv(bool a,const nn::detail::ConvNdOptions<D>& o) {
 K x=KDICT; nn::detail::ConvNdOptions<D> d(o.in_channels(),o.out_channels(),o.kernel_size());
 bool t=o.transposed();
 OPTION(x, in,   kj(o.in_channels()));
 OPTION(x, out,  kj(o.out_channels()));
 OPTION(x, size, KEX(o.kernel_size()));
 if(a || (*o.stride()  != *d.stride()))  OPTION(x, stride, KEX(o.stride()));
 convpad<D>(x,a,d.padding(),o.padding());
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
 return x;
}

// --------------------------------------------------------------------------------------
// fold,unfold - set/get size,dilation,padding,stride
// --------------------------------------------------------------------------------------
static nn::FoldOptions fold(K x,J i,Cast c) {
 nn::FoldOptions o(0,0);
 bool out=false,sz=false; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
   switch(j) {
    case 0: o.output_size(exarray<2>(x,i+j,c,Setting::out)); out=true; break;
    case 1: o.kernel_size(exarray<2>(x,i+j,c,Setting::size)); sz=true; break;
    case 2: o.dilation   (exarray<2>(x,i+j,c,Setting::dilate)); break;
    case 3: o.padding    (exarray<2>(x,i+j,c,Setting::pad));    break;
    case 4: o.stride     (exarray<2>(x,i+j,c,Setting::stride)); break;
    default: mpos(x,c,i+j); break;
  }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::out:       o.output_size(exarray<2>(p,c));out=true; break;
   case Setting::size:      o.kernel_size(exarray<2>(p,c)); sz=true; break;
   case Setting::dilate:    o.dilation   (exarray<2>(p,c)); break;
   case Setting::pad:       o.padding    (exarray<2>(p,c)); break;
   case Setting::stride:    o.stride     (exarray<2>(p,c)); break;
   default: mpair(c,p); break;
  }
 TORCH_CHECK(out, msym(c),": no output size given");
 TORCH_CHECK(sz,  msym(c),": no kernel size given");
 return o;
}

static K fold(bool a,const nn::FoldOptions& o) {
 K x=KDICT; nn::FoldOptions d(o.output_size(),o.kernel_size());
 OPTION(x, out,  KEX(o.output_size()));
 OPTION(x, size, KEX(o.kernel_size()));
 if(a || (*o.dilation() != *d.dilation())) OPTION(x, dilate, KEX(o.dilation()));
 if(a || (*o.padding()  != *d.padding()))  OPTION(x, pad,    KEX(o.padding()));
 if(a || (*o.stride()   != *d.stride()))   OPTION(x, stride, KEX(o.stride()));
 return x;
}

static nn::UnfoldOptions unfold(K x,J i,Cast c) {
 nn::UnfoldOptions o(0);
 bool sz=false; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
   switch(j) {
    case 0: o.kernel_size(exarray<2>(x,i+j,c,Setting::size)); sz=true; break;
    case 1: o.dilation   (exarray<2>(x,i+j,c,Setting::dilate)); break;
    case 2: o.padding    (exarray<2>(x,i+j,c,Setting::pad));    break;
    case 3: o.stride     (exarray<2>(x,i+j,c,Setting::stride)); break;
    default: mpos(x,c,i+j); break;
  }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::size:      o.kernel_size(exarray<2>(p,c)); sz=true; break;
   case Setting::dilate:    o.dilation   (exarray<2>(p,c)); break;
   case Setting::pad:       o.padding    (exarray<2>(p,c)); break;
   case Setting::stride:    o.stride     (exarray<2>(p,c)); break;
   default: mpair(c,p); break;
  }
 TORCH_CHECK(sz, msym(c),": no kernel size given");
 return o;
}

static K unfold(bool a,const nn::UnfoldOptions& o) {
 K x=KDICT; nn::UnfoldOptions d(o.kernel_size());
 OPTION(x, size, KEX(o.kernel_size()));
 if(a || (*o.dilation() != *d.dilation())) OPTION(x, dilate, KEX(o.dilation()));
 if(a || (*o.padding()  != *d.padding()))  OPTION(x, pad,    KEX(o.padding()));
 if(a || (*o.stride()   != *d.stride()))   OPTION(x, stride, KEX(o.stride()));
 return x;
}

static K kfold(K x,Cast c) {
 KTRY
  TORCH_CHECK(!x->t, msym(c)," not implemented for ",kname(x->t));
  Tensor r, *t=xten(x,0);
  return kresult(t, c==Cast::fold
       ? fnn::fold  (t ? *t : kput(x,0),   fold(x,1,c))
       : fnn::unfold(t ? *t : kput(x,0), unfold(x,1,c)));
 KCATCH("fold");
}

KAPI   Fold(K x) {return kfold(x, Cast::fold);}
KAPI Unfold(K x) {return kfold(x, Cast::unfold);}

// --------------------------------------------------------------------------------------
// upsample & interpolate - both require same options, interpolate has additional flag
// upsample is implemented as a module in pytorch, interpolate as a function
// --------------------------------------------------------------------------------------
static void upmode(nn::UpsampleOptions& o,S s) {
 switch(emap(s)) {
  case Enum::nearest:   o.mode(torch::kNearest); break;
  case Enum::linear:    o.mode(torch::kLinear); break;
  case Enum::bilinear:  o.mode(torch::kBilinear); break;
  case Enum::bicubic:   o.mode(torch::kBicubic); break;
  case Enum::trilinear: o.mode(torch::kTrilinear); break;
  default: TORCH_ERROR("unrecognized upsample mode: ",s); break;
 }
}

static void upmode(fnn::InterpolateFuncOptions& o,S s) {
 switch(emap(s)) {
  case Enum::nearest:   o.mode(torch::kNearest); break;
  case Enum::linear:    o.mode(torch::kLinear); break;
  case Enum::bilinear:  o.mode(torch::kBilinear); break;
  case Enum::bicubic:   o.mode(torch::kBicubic); break;
  case Enum::trilinear: o.mode(torch::kTrilinear); break;
  case Enum::area:      o.mode(torch::kArea); break;
  default: TORCH_ERROR("unrecognized interpolate mode: ",s); break;
 }
}

// recompute_scale_factor only part of interpolate options, separate fns to handle setting
static void rescale(K x,J i,Cast c,Setting s,nn::UpsampleOptions& o) {
 TORCH_ERROR(msym(c),": up to 4 positional arguments expected, 5th argument unrecognized");
}
static void rescale(K x,J i,Cast c,Setting s,fnn::InterpolateFuncOptions& o) {
 if(xempty(x,i)) o.recompute_scale_factor({}); else o.recompute_scale_factor(mbool(x,i,c,s));
}
static void rescale(Pairs& p,Cast c,nn::UpsampleOptions& o) {
 TORCH_ERROR(msym(c),": rescale is not a recognized option");
}
static void rescale(Pairs& p,Cast c,fnn::InterpolateFuncOptions& o) {
 if(pempty(p)) o.recompute_scale_factor({}); else o.recompute_scale_factor(mbool(p,c));
}

template<typename O>static O upsample(K x,J i,Cast c) {
 O o; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j) {
  switch(j) {
   case 0: if(xempty(x,i+j)) o.size({}); else o.size(mlongs(x,i+j,c,Setting::size)); break;
   case 1: if(xempty(x,i+j)) o.scale_factor({}); else o.scale_factor(mdoubles(x,i+j,c,Setting::scale)); break;
   case 2: upmode(o,code(x,i+j,c,Setting::mode)); break;
   case 3: if(xempty(x,i+j)) o.align_corners({}); else o.align_corners(mbool(x,i+j,c,Setting::align)); break;
   case 4: rescale(x,i+j,c,Setting::rescale,o); break;
   default: TORCH_ERROR(msym(c),": up to ",(c==Cast::upsample ? 4 : 5)," positional arguments expected, ",n," given");
  }
 }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::size:    if(pempty(p)) o.size({}); else o.size(mlongs(p,c)); break;
   case Setting::scale:   if(pempty(p)) o.scale_factor({}); else o.scale_factor(mdoubles(p,c)); break;
   case Setting::mode:    upmode(o,code(p,c)); break;
   case Setting::align:   if(pempty(p)) o.align_corners({}); else o.align_corners(mbool(p,c)); break;
   case Setting::rescale: rescale(p,c,o); break;
   default: mpair(c,p); break;
  }
 if(o.size()         && !(*o.size()).size())         o.size({});
 if(o.scale_factor() && !(*o.scale_factor()).size()) o.scale_factor({});
 TORCH_CHECK(o.size() || o.scale_factor(), msym(c),": no output size or scale factor given");
 TORCH_CHECK(!(o.size() && o.scale_factor()), msym(c),": both output size and scale factor given");
 return o;
}

template<typename O> static K interp(bool a,const O& o) {
 K x=KDICT; O d;
 if(a || o.size())
  OPTION(x, size, o.size() ? ((*o.size()).size()==1 ? kj((*o.size())[0]) : kget(*o.size())) : ktn(0,0));
 if(a || o.scale_factor())
  OPTION(x, scale, o.scale_factor() ? ((*o.scale_factor()).size()==1 ? kf((*o.scale_factor())[0]) : kget(*o.scale_factor())) : ktn(0,0));
 if(a || o.mode().index() != d.mode().index()) OPTION(x, mode,  ks(ESYM(o.mode())));
 if(a || (d.align_corners() != o.align_corners()) ||
         (d.align_corners() == o.align_corners() &&
          o.align_corners() && *o.align_corners() != *d.align_corners()))
  OPTION(x, align, o.align_corners() ? kb(*o.align_corners()) : ktn(0,0));
 return x;
}

static K upsample(bool a,const nn::UpsampleOptions& o) {return interp(a,o);}

static K interpolate(bool a,const fnn::InterpolateFuncOptions& o) {
 K x=interp(a,o);
 OPTION(x, rescale, o.recompute_scale_factor() ? kb(*o.recompute_scale_factor()) : ktn(0,0));
 return x;
}

KAPI kinterpolate(K x) {
 KTRY
  TORCH_CHECK(!x->t, "interpolate not implemented for ",kname(x->t));
  Tensor r, *t=xten(x,0);
  return kresult(t,
                 fnn::interpolate(t ? *t : kput(x,0),
                 upsample<fnn::InterpolateFuncOptions>(x,1,Cast::interpolate)));
 KCATCH("interpolate");
}

// --------------------------------------------------------------------------------------
// drop - create dropout module given probability/set dictionary given module
// --------------------------------------------------------------------------------------
static nn::DropoutOptions drop(K x,J i,Cast c) {
 nn::DropoutOptions o; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
   switch(j) {
    case 0: o.p(mdouble(x,i+j,c,Setting::p)); break;
    case 1: o.inplace(mbool(x,i+j,c,Setting::inplace)); break;
    default: mpos(x,c,i+j); break;
  }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::p: o.p(mdouble(p,c)); break;
   case Setting::inplace: o.inplace(mbool(p,c)); break;
   default: mpair(c,p); break;
  }
 return o;
}

static K drop(bool a,const nn::DropoutOptions& o) {
 K x=KDICT; nn::DropoutOptions d;
 if(a || o.p()       != d.p())       OPTION(x, p,       kf(o.p()));
 if(a || o.inplace() != d.inplace()) OPTION(x, inplace, kb(o.inplace()));
 return x;
}

// --------------------------------------------------------------------------------------
// create embedding/embedding bag module given options:
// embedmode - translate symbol to internal embedding mode (variant)
// embedset - set name/value pairs specific to Embedding vs EmbeddingBag
// embedpair - handle name/value pairs for both types of embedding modules
// embedwt - handle options depending on whether pre-trained weights supplied
// embed, embedbag - process args and return Embedding/EmbeddingBag module
// --------------------------------------------------------------------------------------
static nn::EmbeddingBagMode embedmode(S s) {
 switch(emap(s)) {
  case Enum::sum:  return torch::kSum;
  case Enum::mean: return torch::kMean;
  case Enum::max:  return torch::kMax;
  default: TORCH_ERROR("unrecognized mode for embedding bag: ",s);
 }
}

static void embedset(Cast c,Setting s,Pairs& p,nn::EmbeddingOptions& o) {
 TORCH_ERROR("unrecognized option for ",msym(c),": ",mset(s));
}

static void embedset(Cast c,Setting s,Pairs& p,nn::EmbeddingBagOptions& o) {
 if     (s == Setting::mode)       o.mode(embedmode(code(p,c)));
 else if(s == Setting::lastoffset) o.include_last_offset(mbool(p,c));
 else TORCH_ERROR("unrecognized option for ",msym(c),": ",mset(s));
}

template<typename O> static void embedpair(Cast c,Pairs& p,O& o,Tensor& w,bool &z) {
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::rows:       o.num_embeddings(int64(p,c)); break;
   case Setting::cols:       o.embedding_dim (int64(p,c)); break;
   case Setting::padindex:   o.padding_idx(int64n(p,c)); break;
   case Setting::maxnorm:    o.max_norm(optdouble(p,c)); break;
   case Setting::p:          o.norm_type(mdouble(p,c)); break;
   case Setting::scale:      o.scale_grad_by_freq(mbool(p,c)); break;
   case Setting::sparse:     o.sparse(mbool(p,c)); break;
   case Setting::weight:     if(!pempty(p)) pten(p,w); break;
   case Setting::freeze:     z=mbool(p,c); break;
   case Setting::mode:       embedset(c,Setting::mode,p,o); break;
   case Setting::lastoffset: embedset(c,Setting::lastoffset,p,o); break;
   default: mpair(c,p); break;
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

static nn::Embedding embed(K x,J i,Cast c) {
 bool z=false; Pairs p; Tensor w; J n=xargc(x,i,p);
 nn::EmbeddingOptions o(nj,nj);
 for(J j=0;j<n;++j) {
   switch(j) {
    case 0:
     switch(kK(x)[i+j]->t) {
      case 0:   if(!xten(x,i+j,w)) w=kput(x,i+j); break;
      case -KJ: o.num_embeddings(int64(x,i+j,c,Setting::rows)); break;
      default:  TORCH_ERROR("embed: 1st arg is number of rows or weight matrix");
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
    default: mpos(x,c,i+j); break;
  }
 }
 embedpair(c,p,o,w,z);
 return embedwt<nn::Embedding,nn::EmbeddingOptions>(c,o,w,z);
}

static nn::EmbeddingBag embedbag(K x,J i,Cast c) {
 bool z=false; Pairs p; Tensor w; J n=xargc(x,i,p);
 nn::EmbeddingBagOptions o(nj,nj);
 // allow mode if last arg even if early in sequence
 if(!x->t && n>1 && n<6 && xsym(x,i+n-1))
  n--, o.mode(embedmode(code(x,i+n,c,Setting::mode)));
 for(J j=0;j<n;++j) {
   switch(j) {
    case 0:
     switch(kK(x)[i+j]->t) {
      case 0:   if(!xten(x,i+j,w)) w=kput(x,i+j); break;
      case -KJ: o.num_embeddings(int64(x,i+j,c,Setting::rows)); break;
      default:  TORCH_ERROR("embed: 1st arg is number of rows or weight matrix");
     }
     break;
    case 1: 
     if(w.defined()) z=mbool(x,i+j,c,Setting::freeze);
     else  o.embedding_dim(int64(x,i+j,c,Setting::cols));
     break;
    case 2: o.max_norm(optdouble(x,i+j,c,Setting::maxnorm)); break;
    case 3: o.norm_type(mdouble(x,i+j,c,Setting::p)); break;
    case 4: o.scale_grad_by_freq(mbool(x,i+j,c,Setting::scale)); break;
    case 5: o.mode(embedmode(code(x,i+j,c,Setting::mode))); break;
    case 6: o.sparse(mbool(x,i+j,c,Setting::sparse)); break;
    case 7: o.padding_idx(int64n(x,i+j,c,Setting::padindex)); break;
    default: mpos(x,c,i+j); break;
  }
 }
 embedpair(c,p,o,w,z);
 return embedwt<nn::EmbeddingBag,nn::EmbeddingBagOptions>(c,o,w,z);
}

// -----------------------------------------------------------------------------------------
// retrieve settings from existing Embedding/EmbeddingBag:
// embedget - templated fucntion to retrieve options specific to Embedding or EmbeddingBag
// embed - templated function which gets options and initial optional weights
// -----------------------------------------------------------------------------------------
static void embedget(bool a,bool b,K x,Cast c,Setting s,const nn::EmbeddingOptions& o,const nn::EmbeddingOptions& d) {
 if(!b && s == Setting::padindex && (a || o.padding_idx().has_value()))
  OPTION(x, padindex, kj(o.padding_idx() ? o.padding_idx().value() : nj));
}

static void embedget(bool a,bool b,K x,Cast c,Setting s,const nn::EmbeddingBagOptions& o,const nn::EmbeddingBagOptions& d) {
 if(s == Setting::mode && (a || o.mode().index() != d.mode().index()))
  OPTION(x, mode, ks(ESYM(o.mode())));
 else if(s == Setting::lastoffset && (a || o.include_last_offset() != d.include_last_offset()))
  OPTION(x, lastoffset, kb(o.include_last_offset()));
 else if(b && s == Setting::padindex && (a || o.padding_idx().has_value()))
  OPTION(x, padindex, kj(o.padding_idx() ? o.padding_idx().value() : nj));
}

template<typename O>static K embed(bool a,Cast c,const O& o,const Tensor& w) {
 K x=KDICT; O d(o.num_embeddings(),o.embedding_dim());
 if(o._weight().defined()) {
  OPTION(x, weight, kget(o._weight()));
  OPTION(x, freeze, kb(!w.requires_grad()));
 } else {
  OPTION(x, rows, kj(o.num_embeddings()));
  OPTION(x, cols, kj(o.embedding_dim()));
 }
 embedget(a,false,x,c,Setting::padindex,o,d);   // embedding only
 if(a || o.max_norm().has_value())                         OPTION(x, maxnorm, kf(o.max_norm() ? o.max_norm().value() : nf));
 if(a || o.norm_type()          != d.norm_type())          OPTION(x, p,       kf(o.norm_type()));
 if(a || o.scale_grad_by_freq() != d.scale_grad_by_freq()) OPTION(x, scale,   kb(o.scale_grad_by_freq()));
 embedget(a,true,x,c,Setting::mode,o,d);        // embedding bag only
 if(a || o.sparse()             != d.sparse())             OPTION(x, sparse,  kb(o.sparse()));
 embedget(a,true,x,c,Setting::lastoffset,o,d);  // embedding bag only
 embedget(a,true,x,c,Setting::padindex,o,d);    // embedding bag only
 return x;
}

// --------------------------------------------------------------------------------------
// linear - parse/retrieve args, invoke functional form
// --------------------------------------------------------------------------------------
static nn::LinearOptions linear(K x,J i,Cast c) {
 bool b=true; int64_t in=nj,out=nj; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j) {
   switch(j) {
    case 0:  in=int64(x,i+j,c,Setting::in);   break;
    case 1: out=int64(x,i+j,c,Setting::out);  break;
    case 2:   b=mbool(x,i+j,c,Setting::bias); break;
    default: mpos(x,c,i+j); break;
  }
 }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::in:   in=int64(p,c); break;
   case Setting::out:  out=int64(p,c); break;
   case Setting::bias: b=mbool(p,c); break;
   default: mpair(c,p); break;
  }
 TORCH_CHECK(in>0,  msym(c), ": positive input size required");
 TORCH_CHECK(out>0, msym(c), ": positive output size required");
 return nn::LinearOptions(in,out).bias(b);
}

static K linear(bool a,const nn::LinearOptions& o) {
 K x=KDICT; nn::LinearOptions d(o.in_features(),o.out_features());
 OPTION(x, in,  kj(o.in_features()));
 OPTION(x, out, kj(o.out_features()));
 if(a || (o.bias() != d.bias())) OPTION(x, bias, kb(o.bias()));
 return x;
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
static nn::BilinearOptions bilinear(K x,J i,Cast c) {
 bool b=true; int64_t in1=nj,in2=nj,out=nj; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j) {
   switch(j) {
    case 0: in1=int64(x,i+j,c,Setting::in1);   break;
    case 1: in2=int64(x,i+j,c,Setting::in2);   break;
    case 2: out=int64(x,i+j,c,Setting::out);  break;
    case 3:   b=mbool(x,i+j,c,Setting::bias); break;
    default: mpos(x,c,i+j); break;
  }
 }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::in1:  in1=int64(p,c); break;
   case Setting::in2:  in2=int64(p,c); break;
   case Setting::out:  out=int64(p,c); break;
   case Setting::bias: b=mbool(p,c); break;
   default: mpair(c,p); break;
  }
 TORCH_CHECK(in1>0 && in2>0, msym(c), ": positive input sizes required");
 TORCH_CHECK(out>0,          msym(c), ": positive output size required");
 return nn::BilinearOptions(in1,in2,out).bias(b);
}

static K bilinear(bool a,const nn::BilinearOptions& o) {
 K x=KDICT; nn::BilinearOptions d(o.in1_features(),o.in2_features(),o.out_features());
 OPTION(x, in1,  kj(o.in1_features()));
 OPTION(x, in2,  kj(o.in2_features()));
 OPTION(x, out, kj(o.out_features()));
 if(a || (o.bias() != d.bias())) OPTION(x, bias, kb(o.bias()));
 return x;
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
//     - rnn accepts non-linear function specification: `tanh or `relu
//       gru & lstm don't have that option, so templates/overloading used for fn setting
// --------------------------------------------------------------------------------------
template<typename O> static void rnnfn(O& o,Cast c,S s) {
 TORCH_ERROR(msym(c),": no non-linear function required (RNN only)");
}

static void rnnfn(nn::RNNOptions& o,Cast c,S s) {
 switch(emap(s)) {
  case Enum::tanh:   o.nonlinearity(torch::kTanh); break;
  case Enum::relu:   o.nonlinearity(torch::kReLU); break;
  default: TORCH_ERROR("unrecognized RNN fn: ",s); break;
 }
}

template<typename O> static void rnnpair(Cast c,Pairs& p,O& o) {
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::in:          o.input_size(int64(p,c)); break;
   case Setting::hidden:      o.hidden_size(int64(p,c)); break;
   case Setting::layers:      o.num_layers(int64(p,c)); break;
   case Setting::fn:          rnnfn(o,c,c==Cast::rnn ? code(p,c) : nullptr); break;
   case Setting::bias:        o.bias(mbool(p,c)); break;
   case Setting::batchfirst:  o.batch_first(mbool(p,c)); break;
   case Setting::dropout:     o.dropout(mdouble(p,c)); break;
   case Setting::bi:          o.bidirectional(mbool(p,c)); break;
   default: mpair(c,p); break;
  }
 TORCH_CHECK(o.hidden_size()>0, msym(c), ": hidden size should be greater than zero");
}

static nn::RNNOptions rnn(K x,J i,Cast c) {
 nn::RNNOptions o(0,0); Pairs p; Tensor w; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
  switch(j) {
   case 0: o.input_size (int64(x,i+j,c,Setting::in)); break;
   case 1: o.hidden_size(int64(x,i+j,c,Setting::hidden)); break;
   case 2: o.num_layers (int64(x,i+j,c,Setting::layers)); break;
   case 3: rnnfn(o,c,code(x,i+j,c,Setting::fn)); break;
   case 4: o.bias(mbool(x,i+j,c,Setting::bias)); break;
   case 5: o.batch_first(mbool(x,i+j,c,Setting::batchfirst)); break;
   case 6: o.dropout(mdouble(x,i+j,c,Setting::dropout)); break;
   case 7: o.bidirectional(mbool(x,i+j,c,Setting::bi)); break;
   default: mpos(x,c,i+j); break;
  }
 rnnpair(c,p,o);
 return o;
}

template<typename O> static O rnn(K x,J i,Cast c) {
 O o(0,0); Pairs p; Tensor w; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
  switch(j) {
   case 0: o.input_size (int64(x,i+j,c,Setting::in)); break;
   case 1: o.hidden_size(int64(x,i+j,c,Setting::hidden)); break;
   case 2: o.num_layers (int64(x,i+j,c,Setting::layers)); break;
   case 3: o.bias(mbool(x,i+j,c,Setting::bias)); break;
   case 4: o.batch_first(mbool(x,i+j,c,Setting::batchfirst)); break;
   case 5: o.dropout(mdouble(x,i+j,c,Setting::dropout)); break;
   case 6: o.bidirectional(mbool(x,i+j,c,Setting::bi)); break;
   default: mpos(x,c,i+j); break;
  }
 rnnpair(c,p,o);
 return o;
}
 
static S rnnfn(const nn::RNNOptions& o) {return ESYM(o.nonlinearity());}
template<typename O>static S rnnfn(const O& o) {return nullptr;}

template<typename O> static K rnn(bool a,const O& o) {
 K x=KDICT; O d(o.input_size(),o.hidden_size()); S s=rnnfn(o);
 OPTION(x, in,     kj(o.input_size()));
 OPTION(x, hidden, kj(o.hidden_size()));
 if(a || (o.num_layers()    != d.num_layers()))   OPTION(x, layers,     kj(o.num_layers()));
 if((a && s) || s           != rnnfn(d))          OPTION(x, fn,         ks(s));
 if(a || (o.bias()          != d.bias()))         OPTION(x, bias,       kb(o.bias()));
 if(a || (o.batch_first()   != d.batch_first()))  OPTION(x, batchfirst, kb(o.batch_first()));
 if(a || (o.dropout()       != d.dropout()))      OPTION(x, dropout,    kf(o.dropout()));
 if(a || (o.bidirectional() != d.bidirectional()))OPTION(x, bi,         kb(o.bidirectional()));
 return x;
}

// -----------------------------------------------------------------------------------
// recur - container layer that applies sequentials to input & output of RNN/GRU/LSTM
//       - functions to parse options (currently only true/false flag for detach)
// -----------------------------------------------------------------------------------
static RecurOptions recur(K x,J i,Cast c) {
 RecurOptions o; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
  switch(j) {
   case 0: o.detach(mbool(x,i+j,c,Setting::detach)); break;
   default: TORCH_ERROR(msym(c),": 1 positional arg(detach flag) expected, ",n," supplied");
  }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::detach: o.detach(mbool(p,c)); break;
   default: mpair(c,p); break;
  }
 return o;
}

static K recur(bool a,const RecurOptions& o) {
 K x=KDICT;
 if(a || o.detach() != RecurOptions().detach()) OPTION(x, detach, kb(o.detach()));
 return x;
}

// ----------------------------------------------------------------------------------
//  maxpool - process args, return dictionary of options, call functional form
// ----------------------------------------------------------------------------------
template<size_t D> static nn::MaxPoolOptions<D> maxpool(K x,J i,Cast c) {
 nn::MaxPoolOptions<D> o(0);
 bool sz=false,st=false; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j) {
   switch(j) {
    case 0: o.kernel_size(exarray<D>(x,i+j,c,Setting::size));    sz=true; break;
    case 1: o.stride     (exarray<D>(x,i+j,c,Setting::stride));  st=true; break;
    case 2: o.padding    (exarray<D>(x,i+j,c,Setting::pad));     break;
    case 3: o.dilation   (exarray<D>(x,i+j,c,Setting::dilate));  break;
    case 4: o.ceil_mode  (mbool     (x,i+j,c,Setting::ceiling)); break;
    default: mpos(x,c,i+j); break;
  }
 }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::size:    o.kernel_size(exarray<D>(p,c)); sz=true; break;
   case Setting::stride:  o.stride     (exarray<D>(p,c)); st=true; break;
   case Setting::pad:     o.padding    (exarray<D>(p,c)); break;
   case Setting::dilate:  o.dilation   (exarray<D>(p,c)); break;
   case Setting::ceiling: o.ceil_mode  (mbool(p,c)); break;
   default: mpair(c,p); break;
  }
 TORCH_CHECK(sz, msym(c),": no kernel size given");
 if(!st) o.stride(o.kernel_size());
 return o;
}

template<typename O> static K maxpool(bool a,const O& o) {
 K x=KDICT; O d(o.kernel_size());
 OPTION(x, size, KEX(o.kernel_size()));
 if(a || *o.stride()   != *d.stride())   OPTION(x, stride,  KEX(o.stride()));
 if(a || *o.padding()  != *d.padding())  OPTION(x, pad,     KEX(o.padding()));
 if(a || *o.dilation() != *d.dilation()) OPTION(x, dilate,  KEX(o.dilation()));
 if(a || o.ceil_mode() != d.ceil_mode()) OPTION(x, ceiling, kb(o.ceil_mode()));
 return x;
}

// ----------------------------------------------------------------------------------
//  avgpool - process args, return dictionary of options, call functional form
// ----------------------------------------------------------------------------------
template<size_t D> static nn::AvgPoolOptions<D> avgpool(K x,J i,Cast c) {
 nn::AvgPoolOptions<D> o(0);
 bool sz=false,st=false; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j) {
   switch(j) {
    case 0: o.kernel_size      (exarray<D>(x,i+j,c,Setting::size));   sz=true; break;
    case 1: o.stride           (exarray<D>(x,i+j,c,Setting::stride)); st=true; break;
    case 2: o.padding          (exarray<D>(x,i+j,c,Setting::pad));      break;
    case 3: o.ceil_mode        (mbool     (x,i+j,c,Setting::ceiling));  break;
    case 4: o.count_include_pad(mbool     (x,i+j,c,Setting::countpad)); break;
    case 5: o.divisor_override (int64n    (x,i+j,c,Setting::divisor));  break;
    default: mpos(x,c,i+j); break;
  }
 }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::size:    o.kernel_size (exarray<D>(p,c)); sz=true; break;
   case Setting::stride:  o.stride      (exarray<D>(p,c)); st=true; break;
   case Setting::pad:     o.padding     (exarray<D>(p,c)); break;
   case Setting::ceiling: o.ceil_mode        (mbool(p,c)); break;
   case Setting::countpad:o.count_include_pad(mbool(p,c)); break;
   case Setting::divisor: o.divisor_override(int64n(p,c)); break;
   default: mpair(c,p); break;
  }
 TORCH_CHECK(sz, msym(c),": no kernel size given");
 if(!st) o.stride(o.kernel_size());
 return o;
}

template<typename O> static K avgpool(bool a,const O& o) {
 K x=KDICT; O d(o.kernel_size());
 OPTION(x, size, KEX(o.kernel_size()));
 if(a || *o.stride()           != *d.stride())           OPTION(x, stride,   KEX(o.stride()));
 if(a || *o.padding()          != *d.padding())          OPTION(x, pad,      KEX(o.padding()));
 if(a || o.ceil_mode()         != d.ceil_mode())         OPTION(x, ceiling,  kb(o.ceil_mode()));
 if(a || o.count_include_pad() != d.count_include_pad()) OPTION(x, countpad, kb(o.count_include_pad()));
 if(a || o.divisor_override().has_value())               OPTION(x, divisor,  kj(o.divisor_override() ? o.divisor_override().value() : nj));
 return x;
}

// ---------------------------------------------------------------------------------------
// adaptive pooling - process args, return dictionary of options, call functional form
// adapt - multiple versions to handle expanding array(1d) vs array of optionals(2,3d)
// ---------------------------------------------------------------------------------------
template<size_t D> static void adapt(ExpandingArray<D>& a,K x,J i,Cast c)        {a=exarray<D>(x,i,c,Setting::size);}
template<size_t D> static void adapt(ExpandingArray<D>& a,const Pairs& p,Cast c) {a=exarray<D>(p,c);}
template<size_t D> static void adapt(Exoptional<D>& a,K x,J i,Cast c)        {a=exoptional<D>(x,i,c,Setting::size);}
template<size_t D> static void adapt(Exoptional<D>& a,const Pairs& p,Cast c) {a=exoptional<D>(p,c);}

template<size_t D> static bool adapt(ExpandingArray<D>& a) {for(const auto &v:*a) if(v != nj) return true; return false;}
template<size_t D> static bool adapt(Exoptional<D>& a)     {for(const auto &v:*a) if(v)       return true; return false;}

template<size_t D,typename T> static T adapt(K x,J i,Cast c) {
 T o(0); bool sz=false; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j) {
   switch(j) {
    case 0: adapt<D>(o.output_size(),x,i+j,c); sz=true; break;
    default: TORCH_ERROR(msym(c),": 1 positional argument expected, ",n," given");
  }
 }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::size: adapt<D>(o.output_size(),p,c); sz=true; break;
   default: mpair(c,p); break;
  }
 TORCH_CHECK(sz, msym(c),": no output size given");
 TORCH_CHECK(adapt(o.output_size()), msym(c),": no output size");
 return o;
}

template<typename O> static K adapt(const O& o) {
 K x=KDICT;
 OPTION(x, size, KEX(o.output_size()));
 return x;
}

// ----------------------------------------------------------------------------------
// fpool - fractional max pooling for 2 & 3d layers
// ----------------------------------------------------------------------------------
template<size_t D> static nn::FractionalMaxPoolOptions<D> fpool(K x,J i,Cast c) {
 nn::FractionalMaxPoolOptions<D> o(0);
 bool e,sz=false; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j) {
   e=xempty(x,i+j);
   switch(j) {
    case 0: o.kernel_size(exarray<D>(x,i+j,c,Setting::size)); sz=true; break;
    case 1: if(e) o.output_size( c10::nullopt); else o.output_size ( exarray<D>(x,i+j,c,Setting::outsize)); break;
    case 2: if(e) o.output_ratio(c10::nullopt); else o.output_ratio(exdouble<D>(x,i+j,c,Setting::ratio));   break;
    default: mpos(x,c,i+j); break;
  }
 }
 while(xpair(p)) {
  e=pempty(p);
  switch(mset(p.k,c)) {
   case Setting::size:    o.kernel_size(exarray<D>(p,c)); sz=true; break;
   case Setting::outsize: if(e) o.output_size (c10::nullopt); else o.output_size(exarray  <D>(p,c)); break;
   case Setting::ratio:   if(e) o.output_ratio(c10::nullopt); else o.output_ratio(exdouble<D>(p,c)); break;
   default: mpair(c,p); break;
  }
 }
 TORCH_CHECK(sz, msym(c), ": no kernel size given");
 TORCH_CHECK(o.output_size()||o.output_ratio(), msym(c), ": no output size or ratio given");
 TORCH_CHECK(!(o.output_size()&&o.output_ratio()), msym(c), ": cannot specify both output size & output ratio");
 return o;
}

template<typename O> static K fpool(bool a,const O& o) {
 K x=KDICT;
 OPTION(x, size, KEX(o.kernel_size()));
 if(a || o.output_size().has_value())    OPTION(x, outsize, o.output_size() ? KEX(o.output_size().value())  : ktn(0,0));
 if(a || o.output_ratio().has_value())   OPTION(x, ratio,   o.output_ratio()? KEX(o.output_ratio().value()) : ktn(0,0));
 return x;
}

// ----------------------------------------------------------------------------------
// lppool - power-average pooling
// ----------------------------------------------------------------------------------
template<size_t D> static nn::LPPoolOptions<D> lppool(K x,J i,Cast c) {
 nn::LPPoolOptions<D> o(0,0);
 bool pw=false,sz=false,st=false; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j) {
   switch(j) {
    case 0: o.norm_type  (mdouble(x,i+j,c,Setting::p));         pw=true; break;
    case 1: o.kernel_size(exarray<D>(x,i+j,c,Setting::size));   sz=true; break;
    case 2: o.stride     (exarray<D>(x,i+j,c,Setting::stride)); st=true; break;
    case 3: o.ceil_mode  (mbool    (x,i+j,c,Setting::ceiling)); break;
    default: mpos(x,c,i+j); break;
  }
 }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::p:       o.norm_type  (mdouble   (p,c)); pw=true; break;
   case Setting::size:    o.kernel_size(exarray<D>(p,c)); sz=true; break;
   case Setting::stride:  o.stride     (exarray<D>(p,c)); st=true; break;
   case Setting::ceiling: o.ceil_mode  (mbool(p,c)); break;
   default: mpair(c,p); break;
  }
 TORCH_CHECK(pw, msym(c),": no power given");
 TORCH_CHECK(sz, msym(c),": no kernel size given");
 if(!st) o.stride(o.kernel_size());
 return o;
}

template<typename O> static K lppool(bool a,const O& o) {
 K x=KDICT; O d(o.norm_type(),o.kernel_size());
 OPTION(x, p,    kf(o.norm_type()));
 OPTION(x, size, KEX(o.kernel_size()));
 if(a || *o.stride()   != *d.stride())   OPTION(x, stride,  KEX(o.stride()));
 if(a || o.ceil_mode() != d.ceil_mode()) OPTION(x, ceiling, kb(o.ceil_mode()));
 return x;
}

// ----------------------------------------------------------------------------------
// functional form of pooling methods:
// ----------------------------------------------------------------------------------
static K pool(K x,Cast c) {
 KTRY
  TORCH_CHECK(!x->t, msym(c)," not implemented for ",kname(x->t));
  Tensor r, *t=xten(x,0);
  switch(c) {
   case Cast::maxpool1d:  r=fnn::max_pool1d(t ? *t : kput(x,0), maxpool<1>(x,1,c)); break;
   case Cast::maxpool2d:  r=fnn::max_pool2d(t ? *t : kput(x,0), maxpool<2>(x,1,c)); break;
   case Cast::maxpool3d:  r=fnn::max_pool3d(t ? *t : kput(x,0), maxpool<3>(x,1,c)); break;
   case Cast::avgpool1d:  r=fnn::avg_pool1d(t ? *t : kput(x,0), avgpool<1>(x,1,c)); break;
   case Cast::avgpool2d:  r=fnn::avg_pool2d(t ? *t : kput(x,0), avgpool<2>(x,1,c)); break;
   case Cast::avgpool3d:  r=fnn::avg_pool3d(t ? *t : kput(x,0), avgpool<3>(x,1,c)); break;
   case Cast::adaptmax1d: r=fnn::adaptive_max_pool1d(t ? *t : kput(x,0), adapt<1,nn::AdaptiveMaxPool1dOptions>(x,1,c)); break;
   case Cast::adaptmax2d: r=fnn::adaptive_max_pool2d(t ? *t : kput(x,0), adapt<2,nn::AdaptiveMaxPool2dOptions>(x,1,c)); break;
   case Cast::adaptmax3d: r=fnn::adaptive_max_pool3d(t ? *t : kput(x,0), adapt<3,nn::AdaptiveMaxPool3dOptions>(x,1,c)); break;
   case Cast::adaptavg1d: r=fnn::adaptive_avg_pool1d(t ? *t : kput(x,0), adapt<1,nn::AdaptiveAvgPool1dOptions>(x,1,c)); break;
   case Cast::adaptavg2d: r=fnn::adaptive_avg_pool2d(t ? *t : kput(x,0), adapt<2,nn::AdaptiveAvgPool2dOptions>(x,1,c)); break;
   case Cast::adaptavg3d: r=fnn::adaptive_avg_pool3d(t ? *t : kput(x,0), adapt<3,nn::AdaptiveAvgPool3dOptions>(x,1,c)); break;
   case Cast::fmaxpool2d: r=fnn::fractional_max_pool2d(t ? *t : kput(x,0), fpool<2>(x,1,c)); break;
   case Cast::fmaxpool3d: r=fnn::fractional_max_pool3d(t ? *t : kput(x,0), fpool<3>(x,1,c)); break;
   case Cast::lppool1d:   r=fnn::lp_pool1d(t ? *t : kput(x,0), lppool<1>(x,1,c)); break;
   case Cast::lppool2d:   r=fnn::lp_pool2d(t ? *t : kput(x,0), lppool<2>(x,1,c)); break;
   default: TORCH_ERROR("unrecognized pooling function");
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
static void padmode(fnn::PadFuncOptions& o,S s) {
 switch(emap(s)) {
  case Enum::constant:  o.mode(torch::kConstant); break;
  case Enum::reflect:   o.mode(torch::kReflect); break;
  case Enum::replicate: o.mode(torch::kReplicate); break;
  case Enum::circular:  o.mode(torch::kCircular); break;
  default: TORCH_ERROR("unrecognized padding mode: ",s); break;
 }
}

static fnn::PadFuncOptions pad(K x,J i,Cast c) {
 fnn::PadFuncOptions o({}); S s; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
  switch(j) {
   case 0: o.pad(mlongs(x,i+j,c,Setting::pad)); break;
   case 1:
    if(xsym(x,i+j,s)) padmode(o,s);
    else if(n==2)     o.value(mdouble(x,i+j,c,Setting::value));
    else TORCH_ERROR("pad: unrecognized 2nd arg, expecting mode or value");
    break;
   case 2: o.value(mdouble(x,i+j,c,Setting::value)); break;
   default: mpos(x,c,i+j); break;
  }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::pad:   o.pad(mlongs(p,c)); break;
   case Setting::mode:  padmode(o,code(p,c)); break;
   case Setting::value: o.value(mdouble(p,c)); break;
   default: mpair(c,p); break;
  }
 n=o.pad().size();
 TORCH_CHECK(n>0 && !(n % 2), msym(c),": ",n," pad size(s) given, expecting pairs for left,right or left,right,top,bottom.. etc");
 return o;
}

static K pad(bool a,const fnn::PadFuncOptions& o) {
 K x=KDICT; const fnn::PadFuncOptions d({});
 OPTION(x, pad, klist(o.pad().size(),o.pad().data()));
 if(a || o.mode().index() != d.mode().index()) OPTION(x, mode,  ks(ESYM(o.mode())));
 if(a || o.value()        != d.value())        OPTION(x, value, kf(o.value()));
 return x;
}

KAPI kpad(K x) {
 KTRY
  TORCH_CHECK(!x->t, "pad not implemented for ",kname(x->t));
  Tensor r, *t=xten(x,0);
  return kresult(t, fnn::pad(t ? *t : kput(x,0), pad(x,1,Cast::pad)));
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
    default: mpos(x,c,i+j); break;
  }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::pad: o.padding(exarray<D*2>(p,c)); sz=true; break;
   case Setting::value: o.value(mdouble(p,c)); break;
   default: mpair(c,p); break;
  }
 TORCH_CHECK(sz, msym(c),": no padding sizes given");
 return o;
}

template<typename O> static K cpad(const O& o) {
 K x=KDICT;
 OPTION(x, pad, KEX(o.padding()));
 OPTION(x, value, kf(o.value()));
 return x;
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
    default: TORCH_ERROR(msym(c),": only 1 positional argument expected, ",n," given");
  }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::pad: o.padding(exarray<D*2>(p,c)); sz=true; break;
   default: mpair(c,p); break;
  }
 TORCH_CHECK(sz, msym(c),": no padding sizes given");
 return o;
}

template<typename O> static K npad(const O& o) {
 K x=KDICT;
 OPTION(x, pad, KEX(o.padding()));
 return x;
}

// ------------------------------------------------------------------------------------
// noarg:  activation fns w'out args,
//         logsigmoid, mish, sigmoid, silu, softsign, tanh, tanhshrink
// ------------------------------------------------------------------------------------
static void noarg(Cast c,K x,J i) {TORCH_CHECK(xnone(x,i), msym(c), ": no arguments expected, ", kstring(x));}

using Ft = Tensor (*)(const Tensor&);
static K noarg(const char* s,Ft f, K x) {
 KTRY
  Tensor *t=xten(x); return kresult(t, f(t ? *t : kput(x)));
 KCATCH(s);
}

KAPI gelu(K x)       {return noarg("gelu",       torch::gelu,        x);}
KAPI logsigmoid(K x) {return noarg("logsigmoid", torch::log_sigmoid, x);}
KAPI mish(K x)       {return noarg("mish",       fnn::mish,          x);}
KAPI silu(K x)       {return noarg("silu",       fnn::silu,          x);}
KAPI softsign(K x)   {return noarg("softsign",   fnn::softsign,      x);}
KAPI tanhshrink(K x) {return noarg("tanhshrink", fnn::tanhshrink,    x);}

// ------------------------------------------------------------------------------------
// activation fns with inplace flag as only arg: relu,relu6,selu
// ------------------------------------------------------------------------------------
static bool inplace(K x,J i,Cast c) {
 bool b=false; Pairs p; J n=xargc(x,i,p);
 if(n)
  TORCH_CHECK(xbool(x,i,b) && n==1, msym(c),": unrecognized option(s), expecting single boolean flag");
 while(xpair(p)) {
  TORCH_CHECK(mset(p.k,c)==Setting::inplace, msym(c),": unrecognized option: ",p.k);
  b=mbool(p,c);
 }
 return b;
}

static K inplace(bool a,bool b) {K x=KDICT; if(a || b) OPTION(x, inplace, kb(b)); return x;}

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
  TORCH_ERROR(msym(c), ": unrecognized positional option(s), expecting alpha, inplace flag, or (alpha;inplace flag)");
 }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::alpha:   o.alpha(mdouble(p,c)); break;
   case Setting::inplace: o.inplace(mbool(p,c)); break;
   default: mpair(c,p); break;
  }
 return o;
}

template<typename O>static K alpha(bool a,const O& o) {
 K x=KDICT; O d;
 if(a || o.alpha()   != d.alpha())   OPTION(x, alpha,   kf(o.alpha()));
 if(a || o.inplace() != d.inplace()) OPTION(x, inplace, kb(o.inplace()));
 return x;
}

// ------------------------------------------------------------------------------------
//  leakyrelu - allow a small positive gradient(slope) when x<0
// ------------------------------------------------------------------------------------
static nn::LeakyReLUOptions slope(K x,J i,Cast c) {
 nn::LeakyReLUOptions o; Pairs p; bool b; J n=xargc(x,i,p);
 if(n==1) {
  if(xbool(x,i,b))
    o.inplace(b);
  else
    o.negative_slope(mdouble(x,i,c,Setting::slope));
 } else if(n==2) {
   o.negative_slope(mdouble(x, i, c, Setting::slope));
   o.inplace(mbool(x,i+1, c, Setting::inplace));
 } else if(n) {
  TORCH_ERROR(msym(c), ": unrecognized positional option(s), expecting slope, inplace flag, or (slope;inplace flag)");
 }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::slope:   o.negative_slope(mdouble(p,c)); break;
   case Setting::inplace: o.inplace(mbool(p,c)); break;
   default: mpair(c,p); break;
  }
 return o;
}

static K slope(bool a,Cast c,const nn::LeakyReLUOptions& o) {
 K x=KDICT; nn::LeakyReLUOptions d;
 if(a || o.negative_slope()   != d.negative_slope()) OPTION(x, slope,   kf(o.negative_slope()));
 if(a || o.inplace()          != d.inplace())        OPTION(x, inplace, kb(o.inplace()));
 return x;
}

// ------------------------------------------------------------------------------------
// hardshrink, softshrink - module/function requires single parm: lambda
// ------------------------------------------------------------------------------------
static double lambda(Cast c) {
 return c==Cast::hardshrink ? nn::HardshrinkOptions().lambda() 
                            : nn::SoftshrinkOptions().lambda();
}

static double lambda(K x,J i,Cast c) {
 double l=lambda(c); Pairs p; J n=xargc(x,i,p);
 if(n==1) l=mdouble(x,i,c,Setting::lambda);
 TORCH_CHECK(n<2,msym(c),": unrecognized positional option(s), expecting lambda, e.g. 0.5");
 while(xpair(p)) {
  TORCH_CHECK(mset(p.k,c)==Setting::lambda,"unrecognized option: ",p.k); l=mdouble(p,c);
 }
 return l;
}

static K lambda(bool a,Cast c,double l) {
 K x=KDICT;
 if(a || l != lambda(c)) OPTION(x,lambda,kf(l));
 return x;
}

// ------------------------------------------------------------------------------------
// cat, glu & softmax,softmax,logsoftmax (modules only) accept single dimension arg
// ------------------------------------------------------------------------------------
static int64_t dim(Cast c) {
 switch(c) {
  case Cast::glu: return nn::GLUOptions().dim();
  case Cast::cat: return CatOptions().dim();
  default:        return nj;
 }
}

static int64_t dim(K x,J i,Cast c) {
 int64_t d=dim(c); Pairs p; J n=xargc(x,i,p);
 if(n==1) d=int64(x,i,c,Setting::dim);
 TORCH_CHECK(n<2, msym(c),": unrecognized positional option(s), expecting single dimension");
 while(xpair(p)) {
  TORCH_CHECK(mset(p.k,c)==Setting::dim,"unrecognized option: ",p.k); d=int64(p,c);
 }
 TORCH_CHECK(d!=nj, msym(c),": no dimension given");
 return d;
}

static K dim(bool a,Cast c,int64_t d) {
 K x=KDICT;
 if(a || d != dim(c)) OPTION(x,dim,kj(d));
 return x;}

// ----------------------------------------------------------------------------------
// onehot - handle option (number of classes) for module or functional implementation
// ----------------------------------------------------------------------------------
static OneHotOptions onehot(K x,J i,Cast c) {  //process arg(s) for number of classes & datatype
 OneHotOptions o; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
  switch(j) {
   case 0: o.num_classes(int64(x,i+j,c,Setting::classes)); break;
   case 1: o.dtype(otype(x,i+j,c,Setting::dtype)); break;
   default: mpos(x,c,i+j); break;
  }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::classes: o.num_classes(int64(p,c)); break;
   case Setting::dtype:   o.dtype(otype(p,c)); break;
   default: mpair(c,p); break;
  }
 TORCH_CHECK(o.num_classes()>-2, msym(c),": number of classes must be nonnegative or set to -1 to derive from input");
 return o;
}

static K onehot(bool a,const OneHotOptions& o) {  //return dictionary for number of classes & optional type
 K x=KDICT;
 OPTION(x,classes,kj(o.num_classes()));
 if(o.dtype()) OPTION(x,dtype,ks(stype(o.dtype().value())));
 return x;
}

static Tensor onehot(const Tensor& t,const OneHotOptions& o) {
 return torch::one_hot(t,o.num_classes()).to(o.dtype() ? o.dtype().value() : torch::kFloat);
}

KAPI Onehot(K x) {  // functional invocation w'additional args for no. of classes,optional datatype
 KTRY
  OneHotOptions o; Tensor *t=xten(x);
  if(t)
   return kten(onehot(*t,o));
  else if((t=xten(x,0)))
   return kten(onehot(*t,onehot(x,1,Cast::onehot)));
  else if(xmixed(x,2))
   return kget(onehot(kput(x,0),onehot(x,1,Cast::onehot)));
  else
   return kget(onehot(kput(x),o));
 KCATCH("onehot");
}

// ----------------------------------------------------------------------------------
// softmax,softmin,logsoftmax: functional form requires dim & optional data type
// softdim: get default dimension from input tensor dimensions (deprecated)
// ----------------------------------------------------------------------------------
static J softdim(size_t d) {return !(d==0 || d==1 || d==3);}

static void softargs(K x,J i,Cast c,J &d,c10::optional<Dtype>& s) { 
 s=c10::nullopt; Pairs p; J n=xargc(x,i,p);
 if(!((n==0 && p.n) || (xlong(x,i,d) && (n==1 || (n==2 && xtype(x,i+1,s))))))
  TORCH_ERROR(msym(c),": unrecognized arg(s), expecting dim or (dim;data type)");
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::dim:   d=int64(p,c); break;
   case Setting::dtype: s=otype(p,c); break;
   default: mpair(c,p); break;
  }
 if(null(d)) 
  TORCH_ERROR("specify the dimension along which ",msym(c)," will be computed");
}

// -----------------------------------------------------------------------------------
// rrelu - randomized leaky relu, functional form has an additional flag for training
// -----------------------------------------------------------------------------------
static void rrelu(K x,J i,Cast c,bool fn,bool& tr,bool& in,double& lo,double& up) {
 Pairs p; J n=xargc(x,i,p); fnn::RReLUFuncOptions o;
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
  switch(mset(p.k,c)) {
   case Setting::lower:   lo=mdouble(p,c); break;
   case Setting::upper:   up=mdouble(p,c); break;
   case Setting::train:   TORCH_CHECK(fn,"rrelu: training flag not set for module"); tr=mbool(p,c);   break;
   case Setting::inplace: in=mbool(p,c);   break;
   default: mpair(c,p); break;
  }
}

// return options for rrelu module
static nn::RReLUOptions rrelu(K x,J i,Cast c) {
 double lo,up; bool in,tr; rrelu(x,i,c,false,tr,in,lo,up);
 return nn::RReLUOptions().lower(lo).upper(up).inplace(in);
}

// retrieve options from rrelu module
static K rrelu(bool a,const nn::RReLUOptions& o) {
 K x=KDICT; nn::RReLUOptions d;
 if(a || d.lower()   != o.lower())   OPTION(x, lower,   kf(o.lower()));
 if(a || d.upper()   != o.upper())   OPTION(x, upper,   kf(o.upper()));
 if(a || d.inplace() != o.inplace()) OPTION(x, inplace, kb(o.inplace()));
 return x;
}

// -----------------------------------------------------------------------------------------
// hardtanh - computationally cheaper version of tanh, straight line at min,max
// -----------------------------------------------------------------------------------------
static nn::HardtanhOptions hardtanh(K x,J i,Cast c) {
 Pairs p; J n=xargc(x,i,p); nn::HardtanhOptions o;
 bool b=o.inplace(); double v1=o.min_val(),v2=o.max_val();
 if(n) {
  TORCH_CHECK((n==1 && (xnum(x,i,v1) || xbool(x,i,b))) ||
              (n==2 &&  xnum(x,i,v1) && (xnum(x,i+1,v2) || xbool(x,i+1,b))) ||
              (n==3 &&  xnum(x,i,v1) &&  xnum(x,i+1,v2) && xbool(x,i+2,b)),
              "hardtanh: unexpected positional arg(s), expects (min;max;inplace flag)");
 }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::min:     v1=mdouble(p,c); break;
   case Setting::max:     v2=mdouble(p,c); break;
   case Setting::inplace: b=mbool(p,c); break;
   default: mpair(c,p); break;
  }
 return o.min_val(v1).max_val(v2).inplace(b);
}

static K hardtanh(bool a,const nn::HardtanhOptions& o) {
 K x=KDICT; nn::HardtanhOptions d;
 if(a || d.min_val() != o.min_val()) OPTION(x, min,     kf(o.min_val()));
 if(a || d.max_val() != o.max_val()) OPTION(x, max,     kf(o.max_val()));
 if(a || d.inplace() != o.inplace()) OPTION(x, inplace, kb(o.inplace()));
 return x;
}

// -----------------------------------------------------------------------------------------
// softplus - smooth approximation to relu, can constrain to always be positive
// -----------------------------------------------------------------------------------------
static nn::SoftplusOptions softplus(K x,J i,Cast c) {
 Pairs p; J n=xargc(x,i,p); nn::SoftplusOptions o; double v1=o.beta(),v2=o.threshold();
 if(n) {
  TORCH_CHECK(xnum(x,i,v1) && (n==1 || (n==2 && xnum(x,i+1,v2))),
              "softplus: unexpected positional arg(s), expects (beta;threshold)");
 }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::beta:      v1=mdouble(p,c); break;
   case Setting::threshold: v2=mdouble(p,c); break;
   default: mpair(c,p); break;
  }
 return o.beta(v1).threshold(v2);
}

static K softplus(bool a,const nn::SoftplusOptions& o) {
 K x=KDICT; nn::SoftplusOptions d;
 if(a || d.beta()      != o.beta())      OPTION(x, beta,      kf(o.beta()));
 if(a || d.threshold() != o.threshold()) OPTION(x, threshold, kf(o.threshold()));
 return x;
}

// ----------------------------------------------------------------------------------------------
// threshold - thresholds each element of input tensor, fns set/get threshold,value,inplace flag
// ----------------------------------------------------------------------------------------------
static nn::ThresholdOptions threshold(K x,J i,Cast c) {
 Pairs p; J n=xargc(x,i,p); bool b=false; double v1=nf,v2=nf;
 if(n) {
  TORCH_CHECK((n==1 && (xnum(x,i,v1) || xbool(x,i,b))) ||
              (n==2 &&  xnum(x,i,v1) && (xnum(x,i+1,v2) || xbool(x,i+1,b))) ||
              (n==3 &&  xnum(x,i,v1) &&  xnum(x,i+1,v2) && xbool(x,i+2,b)),
              "threshold: unexpected positional arg(s), expects (threshold;value;inplace flag)");
 }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::threshold: v1=mdouble(p,c); break;
   case Setting::value:     v2=mdouble(p,c); break;
   case Setting::inplace:   b=mbool(p,c); break;
   default: mpair(c,p); break;
  }
 TORCH_CHECK(v1 == v1 && v2 == v2, "threshold: both threshold level & replacement value must be given");
 return nn::ThresholdOptions(v1,v2).inplace(b);
}

static K threshold(bool a,const nn::ThresholdOptions& o) {
 K x=KDICT;
 OPTION(x, threshold, kf(o.threshold()));
 OPTION(x, value,     kf(o.value()));
 if(a || o.inplace()) OPTION(x, inplace, kb(o.inplace()));
 return x;
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
   case Cast::relu:  r=fnn::relu (t,a ? inplace(x,1,c) : false); break;
   case Cast::relu6: r=fnn::relu6(t,a ? inplace(x,1,c) : false); break;
   case Cast::selu:  r=fnn::selu (t,a ? inplace(x,1,c) : false); break;
   case Cast::elu:   r=fnn::elu(t,alpha<nn::ELUOptions>(a ? x : nullptr,1,c)); break;
   case Cast::celu:  r=fnn::celu(t,alpha<nn::CELUOptions>(a ? x : nullptr,1,c)); break;
   case Cast::leakyrelu: r=fnn::leaky_relu(t,slope(a ? x : nullptr,1,c)); break;
   case Cast::hardshrink: r=torch::hardshrink(t,a ? lambda(x,1,c) : lambda(c)); break;
   case Cast::softshrink: r=torch::softshrink(t,a ? lambda(x,1,c) : lambda(c)); break;
   case Cast::glu:        r=fnn::glu(t,a ? dim(x,1,c) : dim(c)); break;
   case Cast::softmin:
   case Cast::softmax:
   case Cast::logsoftmax: {
    auto d=softdim(t.dim()); c10::optional<Dtype> s; if(a) softargs(x,1,c,d,s);
    switch(c) {
     case Cast::softmin:    r=fnn::detail::softmin(t,d,s); break;
     case Cast::softmax:    r=fnn::detail::softmax(t,d,s); break;
     case Cast::logsoftmax: r=fnn::detail::log_softmax(t,d,s); break;
     default: TORCH_ERROR("unrecognized activation function");
    }
    break;
   }
   case Cast::rrelu: {
    double lo,up; bool in,tr; rrelu(a ? x : nullptr,1,c,false,tr,in,lo,up);
    r=fnn::detail::rrelu(t,lo,up,tr,in);
    break;
   }
   case Cast::hardtanh:  r=fnn::hardtanh (t,  hardtanh(a ? x : nullptr,1,c)); break;
   case Cast::softplus:  r=fnn::softplus (t,  softplus(a ? x : nullptr,1,c)); break;
   case Cast::threshold: r=fnn::threshold(t, threshold(a ? x : nullptr,1,c)); break;
   default: TORCH_ERROR("unrecognized activation function"); break;
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
static nn::PReLUOptions prelu(K x,J i,Cast c) {
 nn::PReLUOptions o; auto m=o.num_parameters();auto w=o.init(); Pairs p; J n=xargc(x,i,p);
 if(n) TORCH_CHECK((n==1 && (xint64(x,i,m) || xdouble(x,i,w))) ||
                   (n==2 &&  xint64(x,i,m) && xdouble(x,i+1,w)),
                   "prelu: expecting 1-2 positional args in,init or (in;init)");
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::in:    m=int64(p,c); break;
   case Setting::init:  w=mdouble(p,c); break;
   default: mpair(c,p); break;
  }
 return o.num_parameters(m).init(w);
}

static K prelu(bool a,const nn::PReLUOptions& o) {
 K x=KDICT; nn::PReLUOptions d;
 if(a || d.num_parameters() != o.num_parameters()) OPTION(x, in,   kj(o.num_parameters()));
 if(a || d.init()           != o.init())           OPTION(x, init, kf(o.init()));
 return x;
}

KAPI Prelu(K x) {
 KTRY
  bool p; Tensor t,w;
  if(!x->t && x->n==2)
   p=xtenarg(x,t,w);
  else if(0<x->t && x->t<98 && x->n==2)
   p=false, t=kput(x), w=t[1], t=t[0];
  else
   TORCH_ERROR("prelu expects 2 args: input & weight, received ",kname(x->t),", count: ",xlen(x));
  return kresult(p, torch::prelu(t,w));
 KCATCH("prelu");
}

// ----------------------------------------------------------------------------------------------------
// distance funtions: could be considered layers or cost functions, so not declared static here
// similar - cosine similarity distance, parse/retrieve optional dimension and epsilon
// pairwise - pairwise distance, parse/retrieve optional power, eps, deep dimension flag
// ----------------------------------------------------------------------------------------------------
nn::CosineSimilarityOptions similar(K x,J i,Cast c) {
 Pairs p; J n=xargc(x,i,p); nn::CosineSimilarityOptions o;
 for(J j=0;j<n;++j)
  switch(j) {
   case 0: o.dim(int64(x,i+j,c,Setting::dim)); break;
   case 1: o.eps(mdouble(x,i+j,c,Setting::eps)); break;
   default: mpos(x,c,i+j); break;
  }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::dim: o.dim(int64(p,c)); break;
   case Setting::eps: o.eps(mdouble(p,c)); break;
   default: mpair(c,p); break;
  }
 return o;
}

// also used w'loss functions in kloss.cpp
K similar(bool a,const nn::CosineSimilarityOptions& o) {
 K x=KDICT; nn::CosineSimilarityOptions d; 
 if(a || (o.dim() != o.dim())) OPTION(x, dim, kj(o.dim()));
 if(a || (o.eps() != d.eps())) OPTION(x, eps, kf(o.eps()));
 return x;
}

nn::PairwiseDistanceOptions pairwise(K x,J i,Cast c) {
 Pairs p; J n=xargc(x,i,p); nn::PairwiseDistanceOptions o;
 for(J j=0;j<n;++j)
  switch(j) {
   case 0: o.p(mdouble(x,i+j,c,Setting::p)); break;
   case 1: o.eps(mdouble(x,i+j,c,Setting::eps)); break;
   case 2: o.keepdim(mbool(x,i+j,c,Setting::keepdim)); break;
   default: mpos(x,c,i+j); break;
  }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::p: o.p(mdouble(p,c)); break;
   case Setting::eps: o.eps(mdouble(p,c)); break;
   case Setting::keepdim: o.keepdim(mbool(p,c)); break;
   default: mpair(c,p); break;
  }
 return o;
}

// also used w'loss functions in kloss.cpp
K pairwise(bool a,const nn::PairwiseDistanceOptions& o) {
 K x=KDICT; nn::PairwiseDistanceOptions d; 
 if(a || (o.p()       != d.p()))       OPTION(x, p,       kf(o.p()));
 if(a || (o.eps()     != d.eps()))     OPTION(x, eps,     kf(o.eps()));
 if(a || (o.keepdim() != d.keepdim())) OPTION(x, keepdim, kb(o.keepdim()));
 return x;
}

// ------------------------------------------------------------------------
// functional form of the distance calculations
// ------------------------------------------------------------------------
static K distance(K x,Cast c) {
 KTRY
  TORCH_CHECK(!x->t, msym(c)," not implemented for ",kname(x->t));
  Tensor r, *a=xten(x,0), *b=xten(x,1);
  switch(c) {
   case Cast::pairwise: r=fnn::pairwise_distance(a ? *a : kput(x,0), b ? *b : kput(x,1), pairwise(x,2,c)); break;
   case Cast::similar:  r=fnn::cosine_similarity(a ? *a : kput(x,0), b ? *b : kput(x,1), similar(x,2,c)); break;
   default: TORCH_ERROR("unrecognized distance function");
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

// -----------------------------------------------------------------------------------------
// flatten - process arg(s) from k and return options
//         - return options used given a flatten module used
//         - call flatten as function given input/tensor and optional start & end dimensions
// -----------------------------------------------------------------------------------------
static nn::FlattenOptions flatten(K x,J i,Cast c,bool f=false);
static nn::FlattenOptions flatten(K x,J i,Cast c,bool f) {
 nn::FlattenOptions o; if(f) o.start_dim(0); Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
   switch(j) {
    case 0: o.start_dim(int64(x,i+j,c,Setting::start));  break;
    case 1: o.end_dim(int64(x,i+j,c,Setting::end));  break;
    default: mpos(x,c,i+j); break;
   }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::start: o.start_dim(int64(p,c)); break;
   case Setting::end:   o.end_dim(int64(p,c)); break;
   default: mpair(c,p); break;
  }
 return o;
}

static K flatten(bool a,const nn::FlattenOptions& o) {
 K x=KDICT; nn::FlattenOptions d;
 if(a || d.start_dim() != o.start_dim()) OPTION(x, start, kj(o.start_dim()));
 if(a || d.end_dim()   != o.end_dim())   OPTION(x, end,   kj(o.end_dim()));
 return x;
}

KAPI Flatten(K x) {  // functional invocation w'different defaults than module(start dim=0 not 1)
 KTRY
  Tensor *t=xten(x);
  if(t) {
   return kten(torch::flatten(*t));
  } else if((t=xten(x,0)) || xmixed(x,2)) {
   auto o=flatten(x,1,Cast::flatten,true);
   return kresult(t, torch::flatten(t ? *t : kput(x,1), o.start_dim(), o.end_dim()));
  } else {
   return kget(torch::flatten(kput(x)));
  }
 KCATCH("flatten");
}

// ----------------------------------------------------------------
// indexselect - get/set dim & tensor index for IndexSelect module
// ----------------------------------------------------------------
static IndexSelectOptions indexselect(K x,J i,Cast c) {
 IndexSelectOptions o(nj,Tensor()); Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
   switch(j) {
    case 0: o.dim(int64(x,i+j,c,Setting::dim));  break;
    case 1: o.ind(ltensor(x,i+j,c,Setting::ind));  break;
    default: mpos(x,c,i+j); break;
   }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::dim: o.dim(int64(p,c)); break;
   case Setting::ind: o.ind(ltensor(p,c)); break;
   default: mpair(c,p); break;
  }
 TORCH_CHECK(o.dim() != nj,     msym(c),": dimension cannot be null");
 TORCH_CHECK(o.ind().defined(), msym(c),": no index defined");
 TORCH_CHECK(o.ind().dim()<2,   msym(c),": expecting scalar index or 1-dim list, given ",o.ind().dim(),"-d tensor"); 
 return o;
}

static K indexselect(bool a,const IndexSelectOptions& o) {
 K x=KDICT;
 OPTION(x, dim,   kj(o.dim()));
 OPTION(x, ind, kget(o.ind()));
 return x;
}

// ----------------------------------------------------------------
// select - get/set dim & scalar index for Select module
// ----------------------------------------------------------------
static SelectOptions select(K x,J i,Cast c) {
 SelectOptions o(nj,nj); Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
   switch(j) {
    case 0: o.dim(int64(x,i+j,c,Setting::dim));  break;
    case 1: o.ind(int64(x,i+j,c,Setting::ind));  break;
    default: mpos(x,c,i+j); break;
   }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::dim: o.dim(int64(p,c)); break;
   case Setting::ind: o.ind(int64(p,c)); break;
   default: mpair(c,p); break;
  }
 TORCH_CHECK(o.dim() != nj, msym(c),": no dimension defined");
 TORCH_CHECK(o.ind() != nj, msym(c),": no index defined");
 return o;
}

static K select(bool a,const SelectOptions& o) {
 K x=KDICT;
 OPTION(x, dim, kj(o.dim()));
 OPTION(x, ind, kj(o.ind()));
 return x;
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
  TORCH_ERROR(msym(c), ": unrecognized positional arg(s), expecting dim, inplace flag, or (dim;inplace flag)");
 }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::dim:     o.dim(int64n(p,c)); break;
   case Setting::inplace: o.inplace(mbool(p,c)); break;
   default: mpair(c,p); break;
  }
 TORCH_CHECK(c==Cast::squeeze || o.dim().has_value(), msym(c),": no dimension given");
 return o;
}

static K squeeze(bool a,const SqueezeOptions& o) {
 K x=KDICT;
 if(o.dim().has_value()) OPTION(x, dim,     kj(o.dim().value()));
 if(a || o.inplace())    OPTION(x, inplace, kb(o.inplace()));
 return x;
}
 
// --------------------------------------------------------------------------------------
// attention - parse/retrieve settings for multi head attention
// --------------------------------------------------------------------------------------
static nn::MultiheadAttentionOptions attention(K x,J i,Cast c) {
 Pairs p; J n=xargc(x,i,p); nn::MultiheadAttentionOptions o(0,0);
 for(J j=0;j<n;++j) {
   switch(j) {
    case 0: o.embed_dim(int64(x,i+j,c,Setting::dim)); break;
    case 1: o.num_heads(int64(x,i+j,c,Setting::heads)); break;
    case 2: o.dropout(mdouble(x,i+j,c,Setting::dropout)); break;
    case 3: o.bias(mbool(x,i+j,c,Setting::bias)); break;
    case 4: o.add_bias_kv(mbool(x,i+j,c,Setting::addbias)); break;
    case 5: o.add_zero_attn(mbool(x,i+j,c,Setting::addzero)); break;
    case 6: o.kdim(int64(x,i+j,c,Setting::kdim)); break;
    case 7: o.vdim(int64(x,i+j,c,Setting::vdim)); break;
    default: mpos(x,c,i+j); break;
  }
 }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::dim:     o.embed_dim(int64(p,c)); break;
   case Setting::heads:   o.num_heads(int64(p,c)); break;
   case Setting::dropout: o.dropout(mdouble(p,c)); break;
   case Setting::bias:    o.bias(mbool(p,c)); break;
   case Setting::addbias: o.add_bias_kv(mbool(p,c)); break;
   case Setting::addzero: o.add_zero_attn(mbool(p,c)); break;
   case Setting::kdim:    o.kdim(int64(p,c)); break;
   case Setting::vdim:    o.vdim(int64(p,c)); break;
   default: mpair(c,p); break;
  }
 TORCH_CHECK(o.embed_dim()>0, msym(c), ": positive embedding dimension required");
 TORCH_CHECK(o.num_heads()>0, msym(c), ": positive number of heads required");
 if(o.kdim()<=0) o.kdim(o.embed_dim());
 if(o.vdim()<=0) o.vdim(o.embed_dim());
 return o;
}

static K attention(bool a,const nn::MultiheadAttentionOptions& o) {
 K x=KDICT; nn::MultiheadAttentionOptions d(o.embed_dim(),o.num_heads());
 OPTION(x, dim,   kj(o.embed_dim()));
 OPTION(x, heads, kj(o.num_heads()));
 if(a || (o.dropout()       != d.dropout()))       OPTION(x, dropout, kf(o.dropout()));
 if(a || (o.bias()          != d.bias()))          OPTION(x, bias,    kb(o.bias()));
 if(a || (o.add_bias_kv()   != d.add_bias_kv()))   OPTION(x, addbias, kb(o.add_bias_kv()));
 if(a || (o.add_zero_attn() != d.add_zero_attn())) OPTION(x, addzero, kb(o.add_zero_attn()));
 if(a || (o.kdim()          != d.kdim()))          OPTION(x, kdim,    kj(o.kdim()));
 if(a || (o.vdim()          != d.vdim()))          OPTION(x, vdim,    kj(o.vdim()));
 return x;
}

// --------------------------------------------------------------------------------------
// encode/decode layers - parse/retrieve settings for multi head attention
// --------------------------------------------------------------------------------------
static nn::TransformerOptions::activation_t codefn(Cast c,S s) {
 switch(emap(s)) {
  case Enum::relu: return torch::kReLU;
  case Enum::gelu: return torch::kGELU;
  default: TORCH_ERROR("unrecognized ", msym(c), " activation fn: ",s); break;
 }
}

template<typename O>static O codelayer(K x,J i,Cast c) {
 TORCH_CHECK(x->t>=0 || x->t==99, msym(c),": unrecognized  or insufficient arg(s), ",kname(x),", ",kstring(x));
 Pairs p; J n=xargc(x,i,p); O o(0,0);
 for(J j=0;j<n;++j) {
  switch(j) {
   case 0: o.d_model(int64(x,i+j,c,Setting::in)); break;
   case 1: o.nhead(int64(x,i+j,c,Setting::heads)); break;
   case 2: o.dim_feedforward(int64(x,i+j,c,Setting::dim)); break;
   case 3: o.dropout(mdouble(x,i+j,c,Setting::dropout)); break;
   case 4: o.activation(codefn(c,code(x,i+j,c,Setting::fn))); break;
   default: mpos(x,c,i+j); break;
  }
 }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::in:      o.d_model(int64(p,c)); break;
   case Setting::heads:   o.nhead(int64(p,c)); break;
   case Setting::dim:     o.dim_feedforward(int64(p,c)); break;
   case Setting::dropout: o.dropout(mdouble(p,c)); break;
   case Setting::fn:      o.activation(codefn(c,code(p,c))); break;
   default: mpair(c,p); break;
  }
 TORCH_CHECK(o.d_model()>0, msym(c), ": positive number of input features required");
 TORCH_CHECK(  o.nhead()>0, msym(c), ": positive number of heads required");
 return o;
}

template<typename O>static K codelayer(bool a,const O& o) {
 K x=KDICT; O d(o.d_model(),o.nhead());
 OPTION(x, in,    kj(o.d_model()));
 OPTION(x, heads, kj(o.nhead()));
 if(a || (o.dim_feedforward()    != d.dim_feedforward()))    OPTION(x, dim,     kj(o.dim_feedforward()));
 if(a || (o.dropout()            != d.dropout()))            OPTION(x, dropout, kf(o.dropout()));
 if(a || (o.activation().index() != d.activation().index())) OPTION(x, fn,      ks(ESYM(o.activation())));
 return x;
}

// ----------------------------------------------------------------------------------------------------
// create transformer encoder/decoder layers:
// codeoff - get offset for processing sum-module arg(s), e.g. (`encodelayer;512;8) -> offset=1
// codelayer - process a created layer module or args to define one
// codenorm - process a created layernorm module or arg(s) to define one
// coder - template for creating encoder/decoder layers with submodule, layer count, optional norm 
// encoder,decoder - invoke template 'coder' with encoder/decoder layer & option types
// ----------------------------------------------------------------------------------------------------
static J codeoff(K x,Cast c) {
 J i=0; S s=nullptr;
 switch(x->t) {
  case 0:   if(xsym(x,0,s)) i=xsym(x,1) ? 2 : 1; break;
  case -KS: s=x->s, i=1; break;
  case KS:  if(x->n) s=kS(x)[0], i=(2<x->n) ? 2 : x->n; break;
  case 99:  i=-1; break;
 }
 if(s) { // module sym not required; if given, check if one of `encoderlayer`decoderlayer`layernorm
  auto m=msym(s);
  TORCH_CHECK(m==Cast::layernorm || m==(c==Cast::encoder ? Cast::encoderlayer : Cast::decoderlayer),
              msym(c), ": unexpected layer '", s, "'");
 }
 return i;
}
 
template<typename L,typename O> static L codelayer(K x,Cast c,std::vector<K>& v) {
 Kmodule *m=xmodule(x); bool e=c == Cast::encoder;
 if(m) {
  auto *l=m->m->as<L>();
  TORCH_CHECK(l, msym(c),": expecting ",(e ? "encoding" : "decoding")," layer, given ",mlabel(m)," module");
  v.push_back(x);
  return L(*l);
 } else {
  return L(codelayer<O>(x,codeoff(x,c),e ? Cast::encoderlayer : Cast::decoderlayer));
 }
}

static nn::LayerNorm codenorm(K x,J n,Cast c,std::vector<K>& v) {
 if(x) {
  auto *m=xmodule(x);
  if(m) {
   auto *l=m->m->as<nn::LayerNorm>();
   TORCH_CHECK(l, msym(c),": expecting normalization layer, given ",mlabel(m)," module");
   v.push_back(x);
   return nn::LayerNorm(*l);
  } else {
   return nn::LayerNorm(x->t==-KJ ? nn::LayerNormOptions({x->j}) : layernorm(x,codeoff(x,c),Cast::layernorm));
  }
 } else {
  return nn::LayerNorm(nn::LayerNormOptions({n}));
 }
}

template<typename R,typename L,typename O>static R coder(K x,J i,Cast c) {
 TORCH_CHECK(x->t==0 || x->t==99, msym(c),": unrecognized  or insufficient arg(s), ",kname(x),", ",kstring(x));
 Pairs p; J l=-1,n=xargc(x,i,p); L m1=nullptr; nn::LayerNorm m2=nullptr; std::vector<K> v;
 for(J j=0;j<n;++j) {
  K y=kK(x)[i+j];
  switch(j) {
   case 0: m1=codelayer<L,O>(y,c,v); break;
   case 1: l=int64(x,i+j,c,Setting::layers); break;
   case 2: if(!xempty(y)) m2=codenorm(x,0,c,v); break;
   default: mpos(x,c,i+j); break;
  }
 }
 while(xpair(p)) {
  switch(mset(p.k,c)) {
   case Setting::decoderlayer:
    TORCH_CHECK(c==Cast::decoder, msym(c),": cannot create a decoder layer");
    TORCH_CHECK(p.t>=0, msym(c),": unrecognized arg(s) for decoder layer (",kname(p.t),")");
    m1=codelayer<L,O>(p.v,c,v);
    break;
   case Setting::encoderlayer:
    TORCH_CHECK(c==Cast::encoder, msym(c),": cannot create an encoder layer");
    TORCH_CHECK(p.t>=0, msym(c),": unrecognized arg(s) for encoder layer (",kname(p.t),")");
    m1=codelayer<L,O>(p.v,c,v);
    break;
   case Setting::layers:
    l=int64(p,c);
    break;
   case Setting::layernorm:
    if(!pempty(p)) {
     TORCH_CHECK(p.t==-KJ || p.t>=0, msym(c),": unrecognized arg(s) for layer normalization (",kname(p.t),")");
     m2=codenorm(p.t==-KJ ? nullptr : p.v, p.j, c, v);
    }
    break;
   default: mpair(c,p); break;
  }
 }
 TORCH_CHECK(l>=0, msym(c), ": non-negative number of layers must be defined");
 auto r=m2.is_empty() ? R(m1,l) : R(m1,l).norm(AnyModule(m2));
 if(v.size()) kfree(v);
 return r;
}

static nn::TransformerDecoderOptions decoder(K x,J i,Cast c) {
 return coder<nn::TransformerDecoderOptions,
              nn::TransformerDecoderLayer,  
              nn::TransformerDecoderLayerOptions>(x,i,c);
}  

static nn::TransformerEncoderOptions encoder(K x,J i,Cast c) {
 return coder<nn::TransformerEncoderOptions,
              nn::TransformerEncoderLayer,  
              nn::TransformerEncoderLayerOptions>(x,i,c);
}  

// -----------------------------------------------------------------------------
// codenorm - retrieve dictionary of options in layer norm module if not empty
// coder - templated retrieval of common options of encoder/decoder layer
// decoder,encoder - retrieve options of encoder/decoder layer,layer count,norm
// -----------------------------------------------------------------------------
static K codenorm(bool a,const AnyModule& m) {
 return m.is_empty() ? KDICT : layernorm(a,m.get<nn::LayerNorm>()->options);
}

template<typename O> static void coder(bool a,K x,const O& o) {
 OPTION(x, layers,    kj(o.num_layers()));
 OPTION(x, layernorm, codenorm(a,o.norm()));
}

static K decoder(bool a,const nn::TransformerDecoderOptions& o) {
 K x=KDICT; OPTION(x, decoderlayer, codelayer(a,o.decoder_layer()->options)); coder(a,x,o); return x;
}

static K encoder(bool a,const nn::TransformerEncoderOptions& o) {
 K x=KDICT; OPTION(x, encoderlayer, codelayer(a,o.encoder_layer()->options)); coder(a,x,o); return x;
}

// -------------------------------------------------------------------------------------
// customcoder - create or retrieve options from custom encoder/decoder for transformer
// transformer - create transformer or retrieve options from existing transformer module
// -------------------------------------------------------------------------------------
static AnyModule customcoder(K x,Setting t,std::vector<K>& v) {
 K y; J i; S s,nm; AnyModule a; Kmodule *m=xmodule(x);
 if(m) {
  a=anymodule(m->c,m->m);
  v.push_back(x);
 } else {
  if(xdict(x)) {
   i=-1; s=statemodule(x); nm=statename(x), y=stateoptions(x);
  } else {
   y=x; msyms(y,s,nm); i=argstart(y,nm);
  }
  a=anymodule(y,i,s);
 }
 return a;
}

static nn::TransformerOptions transformer(K x,J i,Cast c) {
 Pairs p; Setting s; J n=xargc(x,i,p); nn::TransformerOptions o; std::vector<K> v;
 for(J j=0;j<n;++j) {
  switch(j) {
   case 0: o.d_model(int64(x,i+j,c,Setting::in)); break;
   case 1: o.nhead(int64(x,i+j,c,Setting::heads)); break;
   case 2: o.num_encoder_layers(int64(x,i+j,c,Setting::elayers)); break;
   case 3: o.num_decoder_layers(int64(x,i+j,c,Setting::dlayers)); break;
   case 4: o.dim_feedforward(int64(x,i+j,c,Setting::dim)); break;
   case 5: o.dropout(mdouble(x,i+j,c,Setting::dropout)); break;
   case 6: o.activation(codefn(c,code(x,i+j,c,Setting::fn))); break;
   case 7: if(!xempty(x,i+j)) o.custom_encoder(customcoder(kK(x)[i+j],Setting::encoder,v)); break;
   case 8: if(!xempty(x,i+j)) o.custom_decoder(customcoder(kK(x)[i+j],Setting::decoder,v)); break;
   default: mpos(x,c,i+j); break;
  }
 }
 while(xpair(p))
  switch((s=mset(p.k,c))) {
   case Setting::in:      o.d_model(int64(p,c)); break;
   case Setting::heads:   o.nhead(int64(p,c)); break;
   case Setting::elayers: o.num_encoder_layers(int64(p,c)); break;
   case Setting::dlayers: o.num_decoder_layers(int64(p,c)); break;
   case Setting::dim:     o.dim_feedforward(int64(p,c)); break;
   case Setting::dropout: o.dropout(mdouble(p,c)); break;
   case Setting::fn:      o.activation(codefn(c,code(p,c))); break;
   case Setting::encoder: 
   case Setting::decoder:
    if(!pempty(p)) {
     TORCH_CHECK(p.t==-KS || p.t>=0, msym(c), ": unrecognized arg(s) for custom ",p.k);
     auto a=p.t==-KS ? anymodule(nullptr,0,p.s) : customcoder(p.v,s,v);
     s==Setting::encoder ?  o.custom_encoder(a) : o.custom_decoder(a);
    }
    break;
   default: mpair(c,p); break;
  }
 TORCH_CHECK(o.d_model()>0, msym(c), ": positive number of input features required");
 TORCH_CHECK(  o.nhead()>0, msym(c), ": positive number of heads required");
 if(v.size()) kfree(v);  // free any allocated submodules
 return o;
}

static K customcoder(bool a,const AnyModule& y) {
 K k=ktn(KS,2),v=ktn(0,2); const Module& m=*y.ptr(); Cast c=mcast(m);
 kS(k)[0]=statekey(State::module);  kK(v)[0]=ks(msym(c));
 kS(k)[1]=statekey(State::options); kK(v)[1]=mopt(a,false,c,m);
 return xD(k,v);
}

static K transformer(bool a,const nn::TransformerOptions& o) {
 K x=KDICT; nn::TransformerOptions d;
 if(a || (o.d_model()            != d.d_model()))            OPTION(x, in,      kj(o.d_model()));
 if(a || (o.nhead()              != d.nhead()))              OPTION(x, heads,   kj(o.nhead()));
 if(a || (o.num_encoder_layers() != d.num_encoder_layers())) OPTION(x, elayers, kj(o.num_encoder_layers()));
 if(a || (o.num_decoder_layers() != d.num_decoder_layers())) OPTION(x, dlayers, kj(o.num_decoder_layers()));
 if(a || (o.dim_feedforward()    != d.dim_feedforward()))    OPTION(x, dim,     kj(o.dim_feedforward()));
 if(a || (o.dropout()            != d.dropout()))            OPTION(x, dropout, kf(o.dropout()));
 if(a || (o.activation().index() != d.activation().index())) OPTION(x, fn,      ks(ESYM(o.activation())));
 if(o.custom_encoder().is_empty()) {
  if(a) OPTION(x, encoder, KDICT);
 } else { 
  OPTION(x, encoder, customcoder(a,o.custom_encoder()));
 }
 if(o.custom_decoder().is_empty()) {
  if(a) OPTION(x, decoder, KDICT);
 } else { 
  OPTION(x, decoder, customcoder(a,o.custom_decoder()));
 }
 return x;
}

// ---------------------------------------------------------------------------
// zscore - set/get mean,stddev & inplace flag for zscore module
// ---------------------------------------------------------------------------
static ZscoreOptions zscore(K x,J i,Cast c) {
 ZscoreOptions o({},{}); Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
   switch(j) {
    case 0: o.mean(ftensor(x,i+j,c,Setting::mean)); break;
    case 1: o.stddev(ftensor(x,i+j,c,Setting::std)); break;
    case 2: o.inplace(mbool(x,i+j, c, Setting::inplace)); break;
    default: mpos(x,c,i+j); break;
   }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::mean:    o.mean(ftensor(p,c)); break;
   case Setting::std:     o.stddev(ftensor(p,c)); break;
   case Setting::inplace: o.inplace(mbool(p,c)); break;
   default: mpair(c,p); break;
  }
 TORCH_CHECK(o.mean().defined()   && o.mean().numel(),   msym(c),": no mean(s) defined");
 TORCH_CHECK(o.stddev().defined() && o.stddev().numel(), msym(c),": no stddev(s) defined");
 return o;
}

static K zscore(bool a,const ZscoreOptions& o) {
 K x=KDICT;
 OPTION(x, mean, kget(o.mean()));
 OPTION(x, std,  kget(o.stddev()));
 if(a || o.inplace()) OPTION(x, inplace, kb(o.inplace()));
 return x;
}

// ----------------------------------------------------------------------------
// zscore - subtract mean and divide by standard deviation
// ----------------------------------------------------------------------------
Tensor zscore_(Tensor& t,const Tensor& m,const Tensor& d) {
 return t.sub_(m.dim()==1 ? m.view({-1,1,1}) : m).div_(d.dim()==1 ? d.view({-1,1,1}) : d);
}

Tensor zscore(const Tensor& t,const Tensor& m,const Tensor& d) {
 return t.sub(m.dim()==1 ? m.view({-1,1,1}) : m).div(d.dim()==1 ? d.view({-1,1,1}) : d);
}

KAPI kzscore(K x) {
 KTRY
  if(x->t) {
   TORCH_CHECK(x->t > 0, "zscore: not implemented for ",kname(x));
   Tensor t=kput(x);
   TORCH_CHECK(t.dim()>0 && t.size(0)==3, "zscore: expecting 3-element list, given ",x->n," element(s)");
   return kget(zscore(t[0],t[1],t[2]));
  } else {
   Tensor *t=xten(x,0); const auto o=zscore(x,1,Cast::zscore);
   if(t && o.inplace()) {
    return zscore_(*t, o.mean(), o.stddev()), (K)0;
   } else {
    return kresult(t, zscore(t ? *t : kput(x,0), o.mean(), o.stddev()));
   }
  }
 KCATCH("zscore");
}

// ---------------------------------------------------------------------------
// padmode - match k symbol to std::variant style enumeration
// rcrop - set/get probability p and dim for random horizontal/vertical flip
// ---------------------------------------------------------------------------
static void padmode(RandomCropOptions& o,S s) {
 switch(emap(s)) {
  case Enum::constant:  o.padmode(torch::kConstant); break;
  case Enum::reflect:   o.padmode(torch::kReflect); break;
  case Enum::replicate: o.padmode(torch::kReplicate); break;
  case Enum::circular:  o.padmode(torch::kCircular); break;
  default: TORCH_ERROR("unrecognized padding mode: ",s); break;
 }
}

static RandomCropOptions rcrop(K x,J i,Cast c) {
 RandomCropOptions o(0); Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
   switch(j) {
    case 0: o.size (exarray<2>(x,i+j,c,Setting::size)); break;
    case 1: o.pad (exarray<4>(x,i+j,c,Setting::pad)); break;
    case 2: padmode(o,code(x,i+j,c,Setting::padmode)); break;
    case 3: o.value(mdouble(x,i+j,c,Setting::value)); break;
    default: mpos(x,c,i+j); break;
   }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::size:    o.size(exarray<2>(p,c)); break;
   case Setting::pad:     o.pad(exarray<4>(p,c)); break;
   case Setting::padmode: padmode(o,code(p,c)); break;
   case Setting::value:   o.value(mdouble(p,c)); break;
   default: mpair(c,p); break;
  }
 TORCH_CHECK(*o.size() != *torch::ExpandingArray<2>(0), msym(c),": positive cropping height & width not given");
 return o;
}

static K rcrop(bool a,const RandomCropOptions& o) {
 K x=KDICT; const RandomCropOptions d(0);
 OPTION(x, size, KEX(o.size()));
 if(a || *d.pad()            != *o.pad())            OPTION(x, pad,     KEX(o.pad()));
 if(a || d.padmode().index() != o.padmode().index()) OPTION(x, padmode, ks(ESYM(o.padmode())));
 if(a || d.value()           != o.value())           OPTION(x, value,   kf(o.value()));
 return x;
}

// ---------------------------------------------------------------------------
// rcrop - perform random crop given size & padding options
// randomcrop - k api function for random cropping
// ---------------------------------------------------------------------------
static Tensor rcrop(const Tensor& t,int64_t h,int64_t w,const Tensor& z) {
 int64_t r=t.size(-2),c=t.size(-1);      // get rows & cols of tensor to be cropped
 if(r==h && c==w) {                      // if crop size matches tensor rows & cols
  return t;                              // return tensor as is
 } else {
  int64_t y=r-h+1,x=c-w+1;               // else set possible range for top left corner
  TORCH_CHECK(x>0 && y>0, "crop: size ",h,"x",w,", exceeds tensor dim(s) ",r,"x",c);
  y=z.random_(y).item().toLong(), x=z.random_(x).item().toLong();
  return t.index({torch::indexing::Ellipsis, torch::indexing::Slice(y,y+h), torch::indexing::Slice(x,x+w)});
 }
}

static Tensor cpad(const Tensor& t,const RandomCropOptions& o) {
 return *o.pad() == *ExpandingArray<4>(0) ? t : fnn::detail::pad(t,o.pad(),o.padmode(),o.value());
}

Tensor rcrop(const Tensor& t,const RandomCropOptions& o,const Tensor& p) {
  return rcrop(cpad(t,o), (*o.size())[0], (*o.size())[1], p);
}

KAPI randomcrop(K x) {
 KTRY
  TORCH_CHECK(!x->t, "randomcrop: not implemented for ",kname(x));
  TORCH_CHECK(x->n>1 && x->n<6, "randomcrop: 2-5 args expected, (input;size;pad;padmode;value), but ",x->n," given");
  Tensor *t=xten(x,0); Tensor p=torch::empty(1, TensorOptions(t ? t->device() : torch::kCPU).dtype(torch::kLong));
  return kresult(t, rcrop(t ? *t : kput(x,0), rcrop(x,1,Cast::randomcrop), p));
 KCATCH("randomcrop");
}

// ---------------------------------------------------------------------------
// rflip - set/get probability p and dim for random horizontal/vertical flip
// ---------------------------------------------------------------------------
static RandomFlipOptions rflip(K x,J i,Cast c) {
 RandomFlipOptions o; Pairs p; J n=xargc(x,i,p);
 for(J j=0;j<n;++j)
   switch(j) {
    case 0: o.p(mdouble(x,i+j,c,Setting::mean)); break;
    case 1: o.dim(int64(x,i+j,c,Setting::dim)); break;
    default: mpos(x,c,i+j); break;
   }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::p:   o.p(mdouble(p,c)); break;
   case Setting::dim: o.dim(int64(p,c)); break;
   default: mpair(c,p); break;
  }
 return o;
}

static K rflip(bool a,const RandomFlipOptions& o) {
 K x=KDICT; const RandomFlipOptions d;
 if(a || d.p()   != o.p())   OPTION(x, p,   kf(o.p()));
 if(a || d.dim() != o.dim()) OPTION(x, dim, kj(o.dim()));
 return x;
}

static Tensor rflip(const Tensor& t,double p,int64_t d) {
 return torch::empty(1,TensorOptions(t.device()).dtype(torch::kDouble)).uniform_(0,1).item<double>()<p ? t.flip(d) : t;
}

KAPI randomflip(K x) {
 KTRY
  TORCH_CHECK(!x->t, "randomflip: not implemented for ",kname(x));
  Tensor *t=xten(x); if(!t) t=xten(x,0);
  if(t || xmixed(x,3)) {
   const auto o=rflip(x,1,Cast::randomflip);
   return kresult(t, rflip(t ? *t : kput(x,0), o.p(), o.dim()));
  } else {
   const RandomFlipOptions o;
   return kget(rflip(kput(x), o.p(), o.dim()));
  }
 KCATCH("randomflip");
}

// ----------------------------------------------------------------------------------------------------
// getsize - get size(s) for expand & reshape
// expand
// reshape
// ----------------------------------------------------------------------------------------------------
static SizeOptions getsize(K x,J i,Cast c) {
 SizeOptions o({}); Pairs p; J n=xargc(x,i,p);
 TORCH_CHECK(n<2, msym(c),": 1 positional argument expected, ",n," given");
 if(n==1) o.size(mlongs(x,i,c,Setting::size));
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::size: o.size(mlongs(p,c)); break;
   default: mpair(c,p); break;
  }
 return o;
}

static K getsize(bool a,const SizeOptions& o) {
 K x=KDICT;
 OPTION(x, size, klist(o.size().size(),o.size().data()));
 return x;
}

// ----------------------------------------------------------------------------
// mcreate - define module from supplied options, return as generic module ptr
// ----------------------------------------------------------------------------
static Moduleptr mcreate(K x,J i,Cast c) {
 switch(c) {
  case Cast::sequential:  noarg(c,x,i); return nn::Sequential().ptr();  //containers
  case Cast::seqnest:     noarg(c,x,i); return SeqNest().ptr();
  case Cast::seqjoin:     noarg(c,x,i); return SeqJoin().ptr();
  case Cast::moduledict:  noarg(c,x,i); return nn::ModuleDict().ptr();
  case Cast::modulelist:  noarg(c,x,i); return nn::ModuleList().ptr();
  case Cast::fork:        noarg(c,x,i); return Fork().ptr();
  case Cast::residual:    noarg(c,x,i); return Residual().ptr();
  case Cast::transform:   noarg(c,x,i); return Transform().ptr();
  case Cast::nbeats:      noarg(c,x,i); return NBeats().ptr();
  case Cast::base:        noarg(c,x,i); return BaseModule().ptr();
  case Cast::parmdict:    return parmdict(x,i); // dictionary can contain parms as options

  case Cast::batchnorm1d:  return nn::BatchNorm1d(batchnorm<nn::BatchNormOptions>(x,i,c)).ptr();
  case Cast::batchnorm2d:  return nn::BatchNorm2d(batchnorm<nn::BatchNormOptions>(x,i,c)).ptr();
  case Cast::batchnorm3d:  return nn::BatchNorm3d(batchnorm<nn::BatchNormOptions>(x,i,c)).ptr();

  case Cast::instancenorm1d:  return nn::InstanceNorm1d(batchnorm<nn::InstanceNormOptions>(x,i,c)).ptr();
  case Cast::instancenorm2d:  return nn::InstanceNorm2d(batchnorm<nn::InstanceNormOptions>(x,i,c)).ptr();
  case Cast::instancenorm3d:  return nn::InstanceNorm3d(batchnorm<nn::InstanceNormOptions>(x,i,c)).ptr();

  case Cast::groupnorm:  return nn::GroupNorm(groupnorm(x,i,c)).ptr();
  case Cast::layernorm:  return nn::LayerNorm(layernorm(x,i,c)).ptr();
  case Cast::localnorm:  return nn::LocalResponseNorm(localnorm<nn::LocalResponseNormOptions>(x,i,c)).ptr();
  case Cast::crossmap2d: return nn::CrossMapLRN2d(localnorm<nn::CrossMapLRN2dOptions>(x,i,c)).ptr();

  case Cast::embed:        return embed(x,i,c).ptr();
  case Cast::embedbag:     return embedbag(x,i,c).ptr();
  case Cast::linear:       return nn::Linear(linear(x,i,c)).ptr();
  case Cast::bilinear:     return nn::Bilinear(bilinear(x,i,c)).ptr();

  case Cast::drop:         return nn::Dropout(drop(x,i,c)).ptr();
  case Cast::drop2d:       return nn::Dropout2d(drop(x,i,c)).ptr();
  case Cast::drop3d:       return nn::Dropout3d(drop(x,i,c)).ptr();
  case Cast::adrop:        return nn::AlphaDropout(drop(x,i,c)).ptr();
  case Cast::fadrop:       return nn::FeatureAlphaDropout(drop(x,i,c)).ptr();

  case Cast::conv1d:       return nn::Conv1d(conv<1>(x,i,c)).ptr();
  case Cast::conv2d:       return nn::Conv2d(conv<2>(x,i,c)).ptr();
  case Cast::conv3d:       return nn::Conv3d(conv<3>(x,i,c)).ptr();

  case Cast::convtranspose1d:  return nn::ConvTranspose1d(convtran<1>(x,i,c)).ptr();
  case Cast::convtranspose2d:  return nn::ConvTranspose2d(convtran<2>(x,i,c)).ptr();
  case Cast::convtranspose3d:  return nn::ConvTranspose3d(convtran<3>(x,i,c)).ptr();

  case Cast::fold:         return nn::Fold(fold(x,i,c)).ptr();
  case Cast::unfold:       return nn::Unfold(unfold(x,i,c)).ptr();
  case Cast::upsample:     return nn::Upsample(upsample<nn::UpsampleOptions>(x,i,c)).ptr();

  case Cast::maxpool1d:    return nn::MaxPool1d(maxpool<1>(x,i,c)).ptr();
  case Cast::maxpool2d:    return nn::MaxPool2d(maxpool<2>(x,i,c)).ptr();
  case Cast::maxpool3d:    return nn::MaxPool3d(maxpool<3>(x,i,c)).ptr();

  case Cast::avgpool1d:    return nn::AvgPool1d(avgpool<1>(x,i,c)).ptr();
  case Cast::avgpool2d:    return nn::AvgPool2d(avgpool<2>(x,i,c)).ptr();
  case Cast::avgpool3d:    return nn::AvgPool3d(avgpool<3>(x,i,c)).ptr();

  case Cast::adaptmax1d:   return nn::AdaptiveMaxPool1d(adapt<1,nn::AdaptiveMaxPool1dOptions>(x,i,c)).ptr();
  case Cast::adaptmax2d:   return nn::AdaptiveMaxPool2d(adapt<2,nn::AdaptiveMaxPool2dOptions>(x,i,c)).ptr();
  case Cast::adaptmax3d:   return nn::AdaptiveMaxPool3d(adapt<3,nn::AdaptiveMaxPool3dOptions>(x,i,c)).ptr();

  case Cast::adaptavg1d:   return nn::AdaptiveAvgPool1d(adapt<1,nn::AdaptiveAvgPool1dOptions>(x,i,c)).ptr();
  case Cast::adaptavg2d:   return nn::AdaptiveAvgPool2d(adapt<2,nn::AdaptiveAvgPool2dOptions>(x,i,c)).ptr();
  case Cast::adaptavg3d:   return nn::AdaptiveAvgPool3d(adapt<3,nn::AdaptiveAvgPool3dOptions>(x,i,c)).ptr();

  case Cast::fmaxpool2d:   return nn::FractionalMaxPool2d(fpool<2>(x,i,c)).ptr();
  case Cast::fmaxpool3d:   return nn::FractionalMaxPool3d(fpool<3>(x,i,c)).ptr();

  case Cast::lppool1d:     return nn::LPPool1d(lppool<1>(x,i,c)).ptr();
  case Cast::lppool2d:     return nn::LPPool2d(lppool<2>(x,i,c)).ptr();

  case Cast::pad:          return Pad(pad(x,i,c)).ptr();
  case Cast::pad1d:        return nn::ConstantPad1d(cpad<1,nn::ConstantPad1dOptions>(x,i,c)).ptr();
  case Cast::pad2d:        return nn::ConstantPad2d(cpad<2,nn::ConstantPad2dOptions>(x,i,c)).ptr();
  case Cast::pad3d:        return nn::ConstantPad3d(cpad<3,nn::ConstantPad3dOptions>(x,i,c)).ptr();
  case Cast::reflect1d:    return nn::ReflectionPad1d(npad<1,nn::ReflectionPad1dOptions>(x,i,c)).ptr();
  case Cast::reflect2d:    return nn::ReflectionPad2d(npad<2,nn::ReflectionPad2dOptions>(x,i,c)).ptr();
  case Cast::replicate1d:  return nn::ReplicationPad1d(npad<1,nn::ReplicationPad1dOptions>(x,i,c)).ptr();
  case Cast::replicate2d:  return nn::ReplicationPad2d(npad<2,nn::ReplicationPad2dOptions>(x,i,c)).ptr();
  case Cast::replicate3d:  return nn::ReplicationPad3d(npad<3,nn::ReplicationPad3dOptions>(x,i,c)).ptr();
  case Cast::zeropad2d:    return nn::ZeroPad2d(npad<2,nn::ZeroPad2dOptions>(x,i,c)).ptr();

  case Cast::attention:    return nn::MultiheadAttention(attention(x,i,c)).ptr();
  case Cast::decoderlayer: return nn::TransformerDecoderLayer(codelayer<nn::TransformerDecoderLayerOptions>(x,i,c)).ptr();
  case Cast::encoderlayer: return nn::TransformerEncoderLayer(codelayer<nn::TransformerEncoderLayerOptions>(x,i,c)).ptr();
  case Cast::decoder:      return nn::TransformerDecoder(decoder(x,i,c)).ptr();
  case Cast::encoder:      return nn::TransformerEncoder(encoder(x,i,c)).ptr();
  case Cast::transformer:  return nn::Transformer(transformer(x,i,c)).ptr();

  case Cast::rnn:          return nn::RNN(rnn(x,i,c)).ptr();
  case Cast::gru:          return nn::GRU(rnn<nn::GRUOptions>(x,i,c)).ptr();
  case Cast::lstm:         return LSTM(rnn<nn::LSTMOptions>(x,i,c)).ptr();
  case Cast::recur:        return Recur(recur(x,i,c)).ptr();

  case Cast::rnnout:       noarg(c,x,i); return RNNOutput().ptr();
  case Cast::gruout:       noarg(c,x,i); return GRUOutput().ptr();
  case Cast::lstmout:      noarg(c,x,i); return LSTMOutput().ptr();

  case Cast::identity:     noarg(c,x,i); return nn::Identity().ptr();
  case Cast::logsigmoid:   noarg(c,x,i); return nn::LogSigmoid().ptr();
  case Cast::sigmoid:      noarg(c,x,i); return nn::Sigmoid().ptr();
  case Cast::silu:         noarg(c,x,i); return nn::SiLU().ptr();
  case Cast::softsign:     noarg(c,x,i); return nn::Softsign().ptr();
  case Cast::softmax2d:    noarg(c,x,i); return nn::Softmax2d().ptr();
  case Cast::tanh:         noarg(c,x,i); return nn::Tanh().ptr();
  case Cast::tanhshrink:   noarg(c,x,i); return nn::Tanhshrink().ptr();
  case Cast::gelu:         noarg(c,x,i); return nn::GELU().ptr();
  case Cast::mish:         noarg(c,x,i); return nn::Mish().ptr();
  case Cast::mul:          noarg(c,x,i); return Mul().ptr();

  case Cast::relu:         return  nn::ReLU(inplace(x,i,c)).ptr();
  case Cast::relu6:        return nn::ReLU6(inplace(x,i,c)).ptr();
  case Cast::selu:         return  nn::SELU(inplace(x,i,c)).ptr();

  case Cast::softmax:      return nn::Softmax(dim(x,i,c)).ptr();
  case Cast::softmin:      return nn::Softmin(dim(x,i,c)).ptr();
  case Cast::logsoftmax:   return nn::LogSoftmax(dim(x,i,c)).ptr();
  case Cast::flatten:      return nn::Flatten(flatten(x,i,c)).ptr();

  case Cast::select:       return Select(select(x,i,c)).ptr();
  case Cast::indexselect:  return IndexSelect(indexselect(x,i,c)).ptr();
  case Cast::squeeze:      return Squeeze(squeeze(x,i,c)).ptr();
  case Cast::unsqueeze:    return Unsqueeze(squeeze(x,i,c)).ptr();
  case Cast::expand:       return Expand(getsize(x,i,c)).ptr();
  case Cast::reshape:      return Reshape(getsize(x,i,c)).ptr();
  case Cast::cat:          return Cat(dim(x,i,c)).ptr();
  case Cast::onehot:       return OneHot(onehot(x,i,c)).ptr();

  case Cast::elu:          return nn::ELU (alpha<nn::ELUOptions> (x,i,c)).ptr();
  case Cast::celu:         return nn::CELU(alpha<nn::CELUOptions>(x,i,c)).ptr();
  case Cast::leakyrelu:    return nn::LeakyReLU(slope(x,i,c)).ptr();
  case Cast::glu:          return nn::GLU(dim(x,i,c)).ptr();
  case Cast::hardshrink:   return nn::Hardshrink(lambda(x,i,c)).ptr();
  case Cast::softshrink:   return nn::Softshrink(lambda(x,i,c)).ptr();
  case Cast::prelu:        return nn::PReLU(prelu(x,i,c)).ptr();
  case Cast::rrelu:        return nn::RReLU(rrelu(x,i,c)).ptr();
  case Cast::hardtanh:     return nn::Hardtanh(hardtanh(x,i,c)).ptr();
  case Cast::softplus:     return nn::Softplus(softplus(x,i,c)).ptr();
  case Cast::threshold:    return nn::Threshold(threshold(x,i,c)).ptr();
  case Cast::pairwise:     return nn::PairwiseDistance(pairwise(x,i,c)).ptr();
  case Cast::similar:      return nn::CosineSimilarity(similar(x,i,c)).ptr();

  case Cast::zscore:       return Zscore(zscore(x,i,c)).ptr();
  case Cast::randomcrop:   return RandomCrop(rcrop(x,i,c)).ptr();
  case Cast::randomflip:   return RandomFlip(rflip(x,i,c)).ptr();
  default:
   if(container(c))
    TORCH_ERROR("cannot create container module: ",msym(c));
   else
    TORCH_ERROR("unrecognized module: cannot create module from unrecognized enumeration ",(I)c);
 }
}

// ----------------------------------------------------------------------------------------------
// anymodule - given generic module ptr, recast to specific type and return type-erased AnyModule
//             2nd form, where function supplies k arg(s), offset, symbol and returns AnyModule
// ----------------------------------------------------------------------------------------------
#define ANYMODULE(x,y) AnyModule(std::dynamic_pointer_cast<x>(y))
#define ANY(x,y) ANYMODULE(x##Impl,y)

static AnyModule anymodule(Cast c,const Moduleptr& m) {
 switch(c) {
  case Cast::adaptavg1d:      return ANY(nn::AdaptiveAvgPool1d, m);
  case Cast::adaptavg2d:      return ANY(nn::AdaptiveAvgPool2d, m);
  case Cast::adaptavg3d:      return ANY(nn::AdaptiveAvgPool3d, m);
  case Cast::adaptmax1d:      return ANY(nn::AdaptiveMaxPool1d, m);
  case Cast::adaptmax2d:      return ANY(nn::AdaptiveMaxPool2d, m);
  case Cast::adaptmax3d:      return ANY(nn::AdaptiveMaxPool3d, m);
  case Cast::adrop:           return ANY(nn::AlphaDropout, m);
  case Cast::attention:       return ANY(nn::MultiheadAttention, m);
  case Cast::avgpool1d:       return ANY(nn::AvgPool1d, m);
  case Cast::avgpool2d:       return ANY(nn::AvgPool2d, m);
  case Cast::avgpool3d:       return ANY(nn::AvgPool3d, m);
  case Cast::base:            return ANY(BaseModule, m);
  case Cast::batchnorm1d:     return ANY(nn::BatchNorm1d, m);
  case Cast::batchnorm2d:     return ANY(nn::BatchNorm2d, m);
  case Cast::batchnorm3d:     return ANY(nn::BatchNorm3d, m);
  case Cast::bilinear:        return ANY(nn::Bilinear, m);
  case Cast::cat:             return ANY(Cat, m);
  case Cast::celu:            return ANY(nn::CELU, m);
  case Cast::conv1d:          return ANY(nn::Conv1d, m);
  case Cast::conv2d:          return ANY(nn::Conv2d, m);
  case Cast::conv3d:          return ANY(nn::Conv3d, m);
  case Cast::convtranspose1d: return ANY(nn::ConvTranspose1d, m);
  case Cast::convtranspose2d: return ANY(nn::ConvTranspose2d, m);
  case Cast::convtranspose3d: return ANY(nn::ConvTranspose3d, m);
  case Cast::crossmap2d:      return ANY(nn::CrossMapLRN2d, m);
  case Cast::decoder:         return ANY(nn::TransformerDecoder, m);
  case Cast::decoderlayer:    return ANY(nn::TransformerDecoderLayer, m);
  case Cast::drop:            return ANY(nn::Dropout, m);
  case Cast::drop2d:          return ANY(nn::Dropout2d, m);
  case Cast::drop3d:          return ANY(nn::Dropout3d, m);
  case Cast::elu:             return ANY(nn::ELU, m);
  case Cast::embed:           return ANY(nn::Embedding, m);
  case Cast::embedbag:        return ANY(nn::EmbeddingBag, m);
  case Cast::encoder:         return ANY(nn::TransformerEncoder, m);
  case Cast::encoderlayer:    return ANY(nn::TransformerEncoderLayer, m);
  case Cast::expand:          return ANY(Expand, m);
  case Cast::fadrop:          return ANY(nn::FeatureAlphaDropout, m);
  case Cast::flatten:         return ANY(nn::Flatten, m);
  case Cast::fmaxpool2d:      return ANY(nn::FractionalMaxPool2d, m);
  case Cast::fmaxpool3d:      return ANY(nn::FractionalMaxPool3d, m);
  case Cast::fold:            return ANY(nn::Fold, m);
  case Cast::fork:            return ANY(Fork, m);
  case Cast::gelu:            return ANY(nn::GELU, m);
  case Cast::glu:             return ANY(nn::GLU, m);
  case Cast::groupnorm:       return ANY(nn::GroupNorm, m);
  case Cast::gru:             return ANY(nn::GRU, m);
  case Cast::gruout:          return ANY(GRUOutput, m);
  case Cast::hardshrink:      return ANY(nn::Hardshrink, m);
  case Cast::hardtanh:        return ANY(nn::Hardtanh, m);
  case Cast::identity:        return ANY(nn::Identity, m);
  case Cast::indexselect:     return ANY(IndexSelect, m);
  case Cast::instancenorm1d:  return ANY(nn::InstanceNorm1d, m);
  case Cast::instancenorm2d:  return ANY(nn::InstanceNorm2d, m);
  case Cast::instancenorm3d:  return ANY(nn::InstanceNorm3d, m);
  case Cast::layernorm:       return ANY(nn::LayerNorm, m);
  case Cast::leakyrelu:       return ANY(nn::LeakyReLU, m);
  case Cast::linear:          return ANY(nn::Linear, m);
  case Cast::localnorm:       return ANY(nn::LocalResponseNorm, m);
  case Cast::logsigmoid:      return ANY(nn::LogSigmoid, m);
  case Cast::logsoftmax:      return ANY(nn::LogSoftmax, m);
  case Cast::lppool1d:        return ANY(nn::LPPool1d, m);
  case Cast::lppool2d:        return ANY(nn::LPPool2d, m);
  case Cast::lstm:            return ANY(LSTM, m);
  case Cast::lstmout:         return ANY(LSTMOutput, m);
  case Cast::maxpool1d:       return ANY(nn::MaxPool1d, m);
  case Cast::maxpool2d:       return ANY(nn::MaxPool2d, m);
  case Cast::maxpool3d:       return ANY(nn::MaxPool3d, m);
  case Cast::mish:            return ANY(nn::Mish, m);
  case Cast::mul:             return ANY(Mul, m);
  case Cast::nbeats:          return ANY(NBeats, m);
  case Cast::onehot:          return ANY(OneHot, m);
  case Cast::pad:             return ANY(Pad, m);
  case Cast::pad1d:           return ANY(nn::ConstantPad1d, m);
  case Cast::pad2d:           return ANY(nn::ConstantPad2d, m);
  case Cast::pad3d:           return ANY(nn::ConstantPad3d, m);
  case Cast::pairwise:        return ANY(nn::PairwiseDistance, m);
  case Cast::prelu:           return ANY(nn::PReLU, m);
  case Cast::randomcrop:      return ANY(RandomCrop, m);
  case Cast::randomflip:      return ANY(RandomFlip, m);
  case Cast::recur:           return ANY(Recur, m);
  case Cast::reflect1d:       return ANY(nn::ReflectionPad1d, m);
  case Cast::reflect2d:       return ANY(nn::ReflectionPad2d, m);
  case Cast::relu:            return ANY(nn::ReLU, m);
  case Cast::relu6:           return ANY(nn::ReLU6, m);
  case Cast::replicate1d:     return ANY(nn::ReplicationPad1d, m);
  case Cast::replicate2d:     return ANY(nn::ReplicationPad2d, m);
  case Cast::replicate3d:     return ANY(nn::ReplicationPad3d, m);
  case Cast::residual:        return ANY(Residual, m);
  case Cast::reshape:         return ANY(Reshape, m);
  case Cast::rnn:             return ANY(nn::RNN, m);
  case Cast::rnnout:          return ANY(RNNOutput, m);
  case Cast::rrelu:           return ANY(nn::RReLU, m);
  case Cast::select:          return ANY(Select, m);
  case Cast::selu:            return ANY(nn::SELU, m);
  case Cast::seqjoin:         return ANY(SeqJoin, m);
  case Cast::seqnest:         return ANY(SeqNest, m);
  case Cast::sigmoid:         return ANY(nn::Sigmoid, m);
  case Cast::silu:            return ANY(nn::SiLU, m);
  case Cast::similar:         return ANY(nn::CosineSimilarity, m);
  case Cast::softmax:         return ANY(nn::Softmax, m);
  case Cast::softmax2d:       return ANY(nn::Softmax2d, m);
  case Cast::softmin:         return ANY(nn::Softmin, m);
  case Cast::softplus:        return ANY(nn::Softplus, m);
  case Cast::softshrink:      return ANY(nn::Softshrink, m);
  case Cast::softsign:        return ANY(nn::Softsign, m);
  case Cast::squeeze:         return ANY(Squeeze, m);
  case Cast::tanh:            return ANY(nn::Tanh, m);
  case Cast::tanhshrink:      return ANY(nn::Tanhshrink, m);
  case Cast::threshold:       return ANY(nn::Threshold, m);
  case Cast::transform:       return ANY(Transform, m);
  case Cast::transformer:     return ANY(nn::Transformer, m);
  case Cast::unfold:          return ANY(nn::Unfold, m);
  case Cast::unsqueeze:       return ANY(Unsqueeze, m);
  case Cast::upsample:        return ANY(nn::Upsample, m);
  case Cast::zeropad2d:       return ANY(nn::ZeroPad2d, m);
  case Cast::zscore:          return ANY(Zscore, m);

  case Cast::interpolate:
  case Cast::normalize:       TORCH_ERROR(msym(c),": unable to create type-erased module, only functional form implemented");
  case Cast::modulelist:
  case Cast::parmdict:        TORCH_ERROR(msym(c),": unable to create type-erased module, no forward method defined");
  case Cast::sequential:      TORCH_ERROR(msym(c),": unable to create type-erased module, forward method uses template");

  default: TORCH_ERROR("can't create type-erased module, unrecognized cast: ",(I)c);
 }
}

static AnyModule anymodule(K x,J i,S s) {
 Cast c=msym(s);
 return anymodule(c, mcreate(x,i,c));
}

// --------------------------------------------------------------------------------------------
// mopt - given enumeration and generic module, return options as k dictionary
// --------------------------------------------------------------------------------------------
static K mopt(bool a,bool b,Cast c,const Module& m) { //a:all options returned if true, else only non-default
 switch(c) {
  case Cast::sequential:      //container modules w'out options
  case Cast::seqnest:
  case Cast::seqjoin:
  case Cast::moduledict:
  case Cast::modulelist:
  case Cast::fork:
  case Cast::residual:
  case Cast::transform:
  case Cast::nbeats:
  case Cast::base:
  case Cast::gruout:          //take output tensor (from tuple of rnn output)
  case Cast::lstmout:
  case Cast::rnnout:
  case Cast::gelu:            //pointwise activation fns w'out options
  case Cast::identity:
  case Cast::logsigmoid:
  case Cast::mish:
  case Cast::mul:
  case Cast::sigmoid:
  case Cast::silu:
  case Cast::softsign:
  case Cast::softmax2d:
  case Cast::tanh:
  case Cast::tanhshrink:
   return KDICT;

  case Cast::parmdict: return b ? KDICT : kget(m.named_parameters());  // return parms as options if no state required
  
  case Cast::batchnorm1d:      return batchnorm(a,m.as<nn::BatchNorm1d>()->options);
  case Cast::batchnorm2d:      return batchnorm(a,m.as<nn::BatchNorm2d>()->options);
  case Cast::batchnorm3d:      return batchnorm(a,m.as<nn::BatchNorm3d>()->options);
  case Cast::instancenorm1d:   return batchnorm(a,m.as<nn::InstanceNorm1d>()->options);
  case Cast::instancenorm2d:   return batchnorm(a,m.as<nn::InstanceNorm2d>()->options);
  case Cast::instancenorm3d:   return batchnorm(a,m.as<nn::InstanceNorm3d>()->options);
  case Cast::groupnorm:        return groupnorm(a,m.as<nn::GroupNorm>()->options);
  case Cast::layernorm:        return layernorm(a,m.as<nn::LayerNorm>()->options);
  case Cast::localnorm:        return localnorm(a,c,m.as<nn::LocalResponseNorm>()->options);
  case Cast::crossmap2d:       return localnorm(a,c,m.as<nn::CrossMapLRN2d>()->options);

  case Cast::embed:    {auto* e=m.as<nn::Embedding>();    return embed(a,c,e->options,e->weight);}
  case Cast::embedbag: {auto* e=m.as<nn::EmbeddingBag>(); return embed(a,c,e->options,e->weight);}

  case Cast::linear:           return linear(a,m.as<nn::Linear>()->options);
  case Cast::bilinear:         return bilinear(a,m.as<nn::Bilinear>()->options);

  case Cast::drop:             return drop(a,m.as<nn::Dropout>()->options);
  case Cast::drop2d:           return drop(a,m.as<nn::Dropout2d>()->options);
  case Cast::drop3d:           return drop(a,m.as<nn::Dropout3d>()->options);
  case Cast::adrop:            return drop(a,m.as<nn::AlphaDropout>()->options);
  case Cast::fadrop:           return drop(a,m.as<nn::FeatureAlphaDropout>()->options);

  case Cast::conv1d:           return conv(a,m.as<nn::Conv1d>()->options);
  case Cast::conv2d:           return conv(a,m.as<nn::Conv2d>()->options);
  case Cast::conv3d:           return conv(a,m.as<nn::Conv3d>()->options);
  case Cast::convtranspose1d:  return conv(a,m.as<nn::ConvTranspose1d>()->options);
  case Cast::convtranspose2d:  return conv(a,m.as<nn::ConvTranspose2d>()->options);
  case Cast::convtranspose3d:  return conv(a,m.as<nn::ConvTranspose3d>()->options);

  case Cast::fold:             return fold(a,m.as<nn::Fold>()->options);
  case Cast::unfold:           return unfold(a,m.as<nn::Unfold>()->options);
  case Cast::upsample:         return upsample(a,m.as<nn::Upsample>()->options);

  case Cast::maxpool1d:        return maxpool(a,m.as<nn::MaxPool1d>()->options);
  case Cast::maxpool2d:        return maxpool(a,m.as<nn::MaxPool2d>()->options);
  case Cast::maxpool3d:        return maxpool(a,m.as<nn::MaxPool3d>()->options);

  case Cast::avgpool1d:        return avgpool(a,m.as<nn::AvgPool1d>()->options);
  case Cast::avgpool2d:        return avgpool(a,m.as<nn::AvgPool2d>()->options);
  case Cast::avgpool3d:        return avgpool(a,m.as<nn::AvgPool3d>()->options);

  case Cast::adaptmax1d:       return adapt(m.as<nn::AdaptiveMaxPool1d>()->options);
  case Cast::adaptmax2d:       return adapt(m.as<nn::AdaptiveMaxPool2d>()->options);
  case Cast::adaptmax3d:       return adapt(m.as<nn::AdaptiveMaxPool3d>()->options);

  case Cast::adaptavg1d:       return adapt(m.as<nn::AdaptiveAvgPool1d>()->options);
  case Cast::adaptavg2d:       return adapt(m.as<nn::AdaptiveAvgPool2d>()->options);
  case Cast::adaptavg3d:       return adapt(m.as<nn::AdaptiveAvgPool3d>()->options);

  case Cast::fmaxpool2d:       return fpool(a,m.as<nn::FractionalMaxPool2d>()->options);
  case Cast::fmaxpool3d:       return fpool(a,m.as<nn::FractionalMaxPool3d>()->options);

  case Cast::lppool1d:         return lppool(a,m.as<nn::LPPool1d>()->options);
  case Cast::lppool2d:         return lppool(a,m.as<nn::LPPool2d>()->options);

  case Cast::pad:              return pad(a,m.as<Pad>()->options);
  case Cast::pad1d:            return cpad(m.as<nn::ConstantPad1d>()->options);
  case Cast::pad2d:            return cpad(m.as<nn::ConstantPad2d>()->options);
  case Cast::pad3d:            return cpad(m.as<nn::ConstantPad3d>()->options);
  case Cast::reflect1d:        return npad(m.as<nn::ReflectionPad1d>()->options);
  case Cast::reflect2d:        return npad(m.as<nn::ReflectionPad2d>()->options);
  case Cast::replicate1d:      return npad(m.as<nn::ReplicationPad1d>()->options);
  case Cast::replicate2d:      return npad(m.as<nn::ReplicationPad2d>()->options);
  case Cast::replicate3d:      return npad(m.as<nn::ReplicationPad3d>()->options);
  case Cast::zeropad2d:        return npad(m.as<nn::ZeroPad2d>()->options);

  case Cast::attention:        return attention(a,m.as<nn::MultiheadAttention>()->options);
  case Cast::encoderlayer:     return codelayer(a,m.as<nn::TransformerEncoderLayer>()->options);
  case Cast::decoderlayer:     return codelayer(a,m.as<nn::TransformerDecoderLayer>()->options);
  case Cast::encoder:          return encoder(a,m.as<nn::TransformerEncoder>()->options);
  case Cast::decoder:          return decoder(a,m.as<nn::TransformerDecoder>()->options);
  case Cast::transformer:      return transformer(a,m.as<nn::Transformer>()->options);

  case Cast::rnn:              return rnn(a,m.as<nn::RNN>()->options);
  case Cast::gru:              return rnn(a,m.as<nn::GRU>()->options);
  case Cast::lstm:             return rnn(a,m.as<LSTM>()->options);
  case Cast::recur:            return recur(a,m.as<Recur>()->options);

  case Cast::relu:             return inplace(a,m.as<nn::ReLU>()->options.inplace());
  case Cast::selu:             return inplace(a,m.as<nn::SELU>()->options.inplace());
  case Cast::relu6:            return inplace(a,m.as<nn::ReLU6>()->options.inplace());

  case Cast::softmax:          return dim(a,c,m.as<nn::Softmax>()->options.dim());
  case Cast::softmin:          return dim(a,c,m.as<nn::Softmin>()->options.dim());
  case Cast::logsoftmax:       return dim(a,c,m.as<nn::LogSoftmax>()->options.dim());
  case Cast::flatten:          return flatten(a,m.as<nn::Flatten>()->options);

  case Cast::select:           return select(a,m.as<Select>()->options);
  case Cast::indexselect:      return indexselect(a,m.as<IndexSelect>()->options);
  case Cast::squeeze:          return squeeze(a,m.as<Squeeze>()->options);
  case Cast::unsqueeze:        return squeeze(a,m.as<Unsqueeze>()->options);
  case Cast::expand:           return getsize(a,m.as<Expand>()->options);
  case Cast::reshape:          return getsize(a,m.as<Reshape>()->options);
  case Cast::cat:              return dim(a,c,m.as<Cat>()->options.dim());
  case Cast::onehot:           return onehot(a,m.as<OneHot>()->options);

  case Cast::elu:              return alpha(a,m.as<nn::ELU>()->options);
  case Cast::celu:             return alpha(a,m.as<nn::CELU>()->options);
  case Cast::leakyrelu:        return slope(a,c,m.as<nn::LeakyReLU>()->options);
  case Cast::glu:              return dim(a,c,m.as<nn::GLU>()->options.dim());
  case Cast::hardshrink:       return lambda(a,c,m.as<nn::Hardshrink>()->options.lambda());
  case Cast::softshrink:       return lambda(a,c,m.as<nn::Softshrink>()->options.lambda());

  case Cast::prelu:            return prelu(a,m.as<nn::PReLU>()->options);
  case Cast::rrelu:            return rrelu(a,m.as<nn::RReLU>()->options);
  case Cast::hardtanh:         return hardtanh(a,m.as<nn::Hardtanh>()->options);
  case Cast::softplus:         return softplus(a,m.as<nn::Softplus>()->options);
  case Cast::threshold:        return threshold(a,m.as<nn::Threshold>()->options);
  case Cast::pairwise:         return pairwise(a,m.as<nn::PairwiseDistance>()->options);
  case Cast::similar:          return similar(a,m.as<nn::CosineSimilarity>()->options);

  case Cast::zscore:           return zscore(a,m.as<Zscore>()->options);
  case Cast::randomcrop:       return rcrop(a,m.as<RandomCrop>()->options);
  case Cast::randomflip:       return rflip(a,m.as<RandomFlip>()->options);

  default: TORCH_ERROR("unrecognized module: ",m.name(),", unable to retrieve options");
 }
}

// ----------------------------------------------------------------------------------
// mparms - set module parms/buffers from k values in dictionary with matching names
//          handles ParameterDict as special case since no set names for parameters
// ----------------------------------------------------------------------------------
static void mparms(Cast c,S s,Module &m,K x,bool p) { // set named parms/buffers in module m from dict x, p true if parms
 if(c==Cast::parmdict) {
  auto *d=m.as<nn::ParameterDict>();
  TORCH_CHECK(d, "unrecognized module, expecting parameter dictionary, given ",m.name(),", unable to restore parms");
  for(const auto& a:kputd(x)) d->insert(a.key(),a.value());
 } else {
  K k=kK(x)[0],v=kK(x)[1]; Tensor V; if(v->t) V=kput(v);
  for(auto &a:p ? m.named_parameters(false) : m.named_buffers(false)) {
   J i=kfind(k,a.key());
   TORCH_CHECK(i>-1, msym(c), ": unable to find ",(p ? " parameter" : " buffer"),": ",a.key());
   Tensor t=v->t ? V[i] : kput(kK(v)[i]);
   if(a.value().defined()) {
    torch::NoGradGuard g;
    TORCH_CHECK(a.value().dtype() == t.dtype(), (s ? s : msym(c)), ": type mismatch, ", a.key(), " is ", a.value().dtype(), ", input is ", t.dtype());
    TORCH_CHECK(a.value().is_same_size(t),      (s ? s : msym(c)), ": size mismatch, ", a.key(), " is ", a.value().sizes(), ", input is ", t.sizes());
    if (a.value().device() != t.device())
     a.value().set_data(t);
    else
     a.value().set_(t);
   } else {
    a.value()=std::move(t);
   }
  }
 }
}

static void mparms(Cast c,Module &m,K p,K f,S s=nullptr);  // s is full module name (see mfind)
static void mparms(Cast c,Module &m,K p,K f,S s) {
 if(p) mparms(c,s,m,p,true);   // if parms dictionary, set module parms from k dictionary
 if(f) mparms(c,s,m,f,false);  // if buffers defined,  set buffers from k dictionary
}

// -----------------------------------------------------------------------------------------
// addany - convert child module to type-erased AnyModule, then add to container
// addseq - check if generic ptr to a Sequential, if so, add, else add as AnyModule
// addmodule - given parent & child module, add allowable combinations, else error
// addparent - create container, add to any previous parent, push on stack
// addchild - add a child layer to existing parent or push single layer to stack
// -----------------------------------------------------------------------------------------
template<typename M> static void addany(M *m,const char *s,const Moduleptr& y) {
 const auto& a=anymodule(mcast(*y),y);
 if(s) m->push_back(s,a); else m->push_back(a);
}

template<typename M> static void addseq(M *m,const char *s,const Moduleptr& y) {
 if(const auto& q=std::dynamic_pointer_cast<torch::nn::SequentialImpl>(y)) {
  if(s) m->push_back(s,nn::Sequential(q)); else m->push_back(nn::Sequential(q));
 } else {
  addany(m,s,y);
 }
}

static void addmodule(Moduleptr& x,const Moduleptr& y) {
 const char* s=mname(*y);
 if(auto *m=x->as<nn::Sequential>())        { addany(m,s,y);
 } else if(auto *m=x->as<SeqNest>())        { addany(m,s,y);
 } else if(auto *m=x->as<SeqJoin>())        { addseq(m,s,y);
 } else if(auto *m=x->as<Fork>())           { addseq(m,s,y);
 } else if(auto *m=x->as<Residual>())       { addseq(m,s,y);
 } else if(auto *m=x->as<Transform>())      { addseq(m,s,y);
 } else if(auto *m=x->as<NBeats>())         { m->push_back(y);
 } else if(auto *m=x->as<Recur>())          { m->push_back(y);
 } else if(auto *m=x->as<nn::ModuleList>()) { m->push_back(y);
 } else if(auto *m=x->as<nn::ModuleDict>()) {
  m->update({{s ? s : c10::to_string(m->children().size()), y}});
 } else if(auto *m=x->as<BaseModule>()) {
  m->register_module(s ? s : c10::to_string(m->children().size()), y);
 } else {
  TORCH_ERROR("unable to add a ", mlabel(y)," module as a child of a ",mlabel(x), " module");
 }
}

static void addname(Module& a,S s) {if(s) mname_(a)=s; else mname_(a)=c10::nullopt;}
 
static void addparent(const Moduleptr& m,Modules& q) {
 if(q.size()) addmodule(q.top(),m);  // add to previous parent, if any
 q.push(m);                          // add new parent container to stack
}

static void addparent(Cast c,S s,Modules& q,K x=nullptr,K y=nullptr,K z=nullptr);
static void addparent(Cast c,S s,Modules& q,K x,K y,K z) {
 TORCH_CHECK(!(c != Cast::parmdict && y && xlen(y)),    msym(c), ": no parameters expected");
 TORCH_CHECK(!(z && xlen(z)),                           msym(c), ": no buffers expected");
 auto m=mcreate(x,argstart(x,s),c); // create generic module ptr from cast, options & offset
 if(y||z) mparms(c,*m,y,z);         // add any supplied parms or buffers
 addname(*m,s);                     // add name if supplied
 addparent(m,q);                    // add to any previous parent, push on stack
}

static void addchild(const Moduleptr& m,Modules& q) {
 if(q.size())
  addmodule(q.top(),m);
 else
  q.push(m);
}

static auto addchild(Cast c,S s,Modules& q,K x,K y=nullptr,K z=nullptr);
static auto addchild(Cast c,S s,Modules& q,K x,K y,K z) {
 auto m=mcreate(x,argstart(x,s),c);   // create generic module ptr from cast, options & offset
 addname(*m,s);                       // add name if supplied
 if(y||z) mparms(c,*m,y,z);           // add any supplied parms or buffers
 addchild(m,q);                       // add to immediate parent container on stack
 return m->modules(false).size();     // return count of all sub-modules created
}

// -------------------------------------------------------------------------------
// msuffix - compare submodule name from newly created module with stored suffix
// mcompare - compare options from two modules, return true if all match exactly
// mfind - match previous state of implicitly defined submodules 
//       - e.g. the self attention layer of an explicitly defined decoder layer
// -------------------------------------------------------------------------------
static bool msuffix(const std::string& x,const std::string& y) {
 return x.size()>=y.size() && !x.compare(x.size()-y.size(),y.size(),y);
}

static bool mcompare(Cast c,const Module& m1,const Module& m2) {
 bool b=false; Cast v=mcast(m1),w=mcast(m2);
 if(v==w) {
  K x=mopt(true,false,v,m1),y=mopt(true,false,w,m2),z;
  z=k(0,(S)"~",x,y,0); b=z->g; r0(z);
 }
 return b;
}

static void mfind(Cast c,J j,S s,Moduleptr& p,K x,K y,K z) {
 TORCH_CHECK(s, "attempting to find ",msym(c)," layer in ",mlabel(p),", but no name given");
 J i=0; bool b=false; 
 for(const auto& a:p->named_modules(std::string(),false)) {
  if(i==j) {
   TORCH_CHECK(msuffix(a.key(),s),"child module mismatch: ",a.key()," does not end with expected suffix '",s,"'");
   auto& m=*a.value();
   TORCH_CHECK(mcompare(c,m,*mcreate(x,argstart(x,s),c)),"child module ",a.key()," mismatch with given options");
   if(y||z) mparms(c,m,y,z,(S)a.key().c_str());   // reset any supplied parms or buffers
   b=true;
   return;
  }
  i++;
 }
 TORCH_CHECK(b, "unable to find ",msym(c),"(",s,") in parent ",mlabel(p));
}

// --------------------------------------------------------------------------------------------
// mdepth - check given depth, must be non-zero if stack populated, no greater than stack size
// mparent - check stack for "parent" - a module with child module(s) that are not user-defined
// mpush - add new parent/child module to network stored in stack of layers
// mpushtable - used when full table format is used to define modules (w'extra submodule rows)
// --------------------------------------------------------------------------------------------
static void mdepth(Cast c,J d,Modules& q) {
 auto n=q.size(); decltype(n) dn=d;  // convert depth to unsigned to be able to compare with stack size
 TORCH_CHECK(dn >=(n ? 1 : 0), msym(c), ": depth ",dn," below min depth of ",n ? 1 : 0);
 TORCH_CHECK(dn <= n,          msym(c), ": depth ",dn," above max depth of ",n);
 while(q.size()>dn) q.pop();
}

static Moduleptr mparent(const Modules& q) {
 Moduleptr m;
 if(q.size()) {
  const auto& p=q.top();          // if module stack has a container at the top
  if(container(p)) {              // check if latest added submodule is a parent
   const auto& c=p->children();   // e.g. decoder, attention, etc.
   return (c.size() && c.back()->children().size()) ? c.back() : nullptr;
  } else {                        // stack of only one non-container, check if parent
   return p->children().size() ? p : nullptr;
  }
 } else {
  return nullptr;
 }
}
   
static Cast mpush(Modules& q,J d,S s,S nm,K x,K y=nullptr,K z=nullptr);
static Cast mpush(Modules& q,J d,S s,S nm,K x,K y,K z) {
 Cast c=msym(s); mdepth(c,d,q);
 if(container(c))
  addparent(c,nm,q,x,y,z);
 else
  addchild(c,nm,q,x,y,z);
 return c;
}

static std::tuple<Cast,J> mpushtable(Modules& q,J j,J d,S s,S nm,K x,K y=nullptr,K z=nullptr);
static std::tuple<Cast,J> mpushtable(Modules& q,J j,J d,S s,S nm,K x,K y,K z) {
 // p defined if module w'children is only member of stack or last module of most recent container
 Moduleptr p=mparent(q);
 if(p && d>(J)(container(q.top()) ? q.size() : 0)) {
  auto c=msym(s); mfind(c,j,nm,p,x,y,z);
  return std::make_tuple(c, ++j);
 } else {
  return std::make_tuple(mpush(q,d,s,nm,x,y,z), 0);
 }
}

static Cast mpush(Modules& q,J d,K x) {S s,nm; msyms(x,s,nm); return mpush(q,d,s,nm,x);}

// -------------------------------------------------------------------------------
// mtree - parse nested tree of layers -- type,name,options -- to build modules
// mdv - parse (depth;value) pair(s) to build module(s)
// mtable - module(s) from table of options & depth, optional name,parms & buffers
// mextend - add a created module to existing module(s) at optional depth
// -------------------------------------------------------------------------------
static Cast mtree(K x,size_t d,Modules& q) {
 K y=x->t || !x->n ? x : kK(x)[0];
 Cast c=mpush(q,d,y);    // get type of overall container module
 if(!x->t)               // process any child modules
  for(J i=1;i<x->n;i++)
   mtree(kK(x)[i],d+1,q);
 return c;
}

static K mtree(K x,J d=0,Kmodule *m=nullptr); // higher-level call, can add to existing module
static K mtree(K x,J d,Kmodule *m) {
 Modules q=mstack(m);
 Cast c=mtree(x,d ? d : q.size(),q);
 return mresult(m,c,q);
}

static Cast mdv(K x,J n,Modules& q) { // process n depth-value pairs, n=-1 for single pair, e.g. (1;(`linear;784;10))
 Cast c,p=Cast::undefined; J d,m=n<0 ? 0 : n; K v;
 for(J i=n<0 ? -1 : 0;i<m;++i) {
  d=dvd(x,i); v=dvv(x,i); c=mpush(q,d,v);
  if(p==Cast::undefined) p=c;
 }
 return p;  // return module enumeration of overall parent container
}

static K mdv(K x,J n,Kmodule *m=nullptr,J d=0,K v=nullptr); // higher-level call, can add to existing module
static K mdv(K x,J n,Kmodule *m,J d,K v) {
 Cast c; Modules q=mstack(m);
 c=v ? mpush(q,d ? d : q.size(),v) : mdv(x,n,q);
 return mresult(m,c,q);
}

static Cast mtable(K x,Modules &q) { // process table/dict w'depth,module,options,parms,buffers
 Cast c,p=Cast::undefined; J j=0,n=x->t==99 ? 1 : xlen(x);
 for(J i=0;i<n;++i) {
  std::tie(c,j)=mpushtable(q, j, statedepth(x,i),   statemodule(x,i), statename(x,i),
                                 stateoptions(x,i), stateparms(x,i),  statebuffers(x,i));
  if(p==Cast::undefined) p=c;
 }
 return p;
}

static K mtable(K x,Kmodule *m=nullptr);  //higher-level call, can also add to existing module if supplied
static K mtable(K x,Kmodule *m) {Modules q=mstack(m); Cast c=mtable(x,q); return mresult(m,c,q);}

static void mextend(Moduleptr& a,Cast c,J d,Modules& q) {
 if(d) mdepth(c,d,q);
 if(container(c))
  addparent(a,q);
 else
  addchild(a,q);
}

static void mextend(Kmodule *x,Kmodule *y,J d=0);
static void mextend(Kmodule *x,Kmodule *y,J d) {
 Modules q=mstack(x);                     //initialize stack of modules
 mextend(y->m,y->c,d ? d : q.size(),q);   //add additional module(s)
}

// --------------------------------------------------------------------------------------------
// mget - extract module options and, optionally, parameters & buffers to k array
// --------------------------------------------------------------------------------------------
static void mget(bool a,bool b,int64_t d,const char* s,bool t,const Module& m,K x) {
 Cast c=mcast(m); K o=mopt(a,b,c,m),*k=kK(x);
 if(!s) s="";
 if(t) {
  ja(&k[0], &d);
  js(&k[1], msym(c));
  js(&k[2], cs(s));
  jk(&k[3], o);
  if(x->n == 6)
   jk(&k[4], kget(m.named_parameters(false))),
   jk(&k[5], kget(m.named_buffers(false)));
  for(const auto& i:m.named_children())
   mget(a,b,d+1,i.key().c_str(),t,*i.value(),x);
 } else {
  TORCH_CHECK(!m.children().size(), msym(c), ": unexpected child module(s)");
  k[0]=kj(d);
  k[1]=ks(msym(c));
  k[2]=ks(cs(s));
  k[3]=o;
  if(x->n == 6)
   k[4]=kget(m.named_parameters(false)),
   k[5]=kget(m.named_buffers(false));
 }
}

K mget(bool a,bool b,const Module& m) {  
// a-true for all options else non-defaults, b-true for full state w'parms & buffers, s-name
 K k=mkeys(b),v=ktn( 0, b ? 6 : 4);  // key,val for depth,module,name,options w'parms & buffers if b
 if(container(m) || m.children().size()) {
  for(J i=0; i<v->n; ++i) kK(v)[i]=ktn(!i ? KJ : (i<3 ? KS : 0), 0);
  mget(a,b,0,mname(m),true,m,v);
  return xT(xD(k,v));
 } else {
  mget(a,b,0,mname(m),false,m,v);
  return xD(k,v);
 }
}

// ------------------------------------------------------------------------------------------
//  main module api function defined in k
// ------------------------------------------------------------------------------------------
KAPI module(K x) {
 KTRY
  bool a=env().alloptions; J d,n; Kmodule *l,*g; Kmodel *m; Kopt* o;
  if((l=xmodule(x)) || (l=xmodule(x,0))) {       // allocated module ptr supplied
   if(x->n==1 || (x->n==2 && xbool(x,1,a))) {    // no other args or boolean flag
    return mget(a,false,*l->m);                  // return module options
   } else if(x->n==2) {                          // else if allocated module & non-boolean arg
    if((g=xmodule(x,1)))                         // if another allocated module
     return mextend(l,g), kfree(x,1), (K)0;      // add to last container module in chain
    else if((n=xdv(x,1)))                        // 2nd arg of depth,value pair(s)
     return mdv(kK(x)[1],n,l);                   // add module(s) specified in depth,value pair(s)
    else if(xstate(x,1))                         // if state dictionary/table detected as 2nd arg
     return mtable(kK(x)[1],l);                  // add definition(s) to existing module(s)
    else                                         // fallback: assume 2nd arg is nested tree spec
     return mtree(kK(x)[1],0,l);                 // add module(s) to last container in existing module
   } else if(x->n==3 && xlong(x,1,d)) {          // else if allocated module & depth given w'3rd arg
    if((g=xmodule(x,2)))                         // if another allocated module
     return mextend(l,g,d), kfree(x,2), (K)0;    // add module at given depth in chain
    else
     return mdv(nullptr,0,l,d,kK(x)[2]);         // add single module definition at indicated depth
   } else {
    TORCH_ERROR("module: ", mlabel(l), " given as 1st arg, but unable to parse remaining arg(s)");
   }
  } else if(xstate(x)) {                         // module table or dictionary supplied
   return mtable(x);
  } else if((m=xmodel(x))) {                     // model ptr supplied, extract module with added reference
   return kmodule(m->mc,m->m);
  } else if((o=xoptim(x))) {                     // optimizer ptr, extract module
   TORCH_CHECK(o->m, "module: no module registered with given optimizer");
   return kmodule(mcast(*o->m),o->m);            // return new k-api handle to this module
  } else if((n=xdv(x))) {                        // depth-value pairs supplied
   return mdv(x,n);
  } else {
   return mtree(x);                              // nested tree representation
  }
 KCATCH("module");
}

// ------------------------------------------------------------------------------------------
//  modulehelp - return a table, via q)module`help, or single dict of options, help`conv2d
// ------------------------------------------------------------------------------------------
K modulehelp(Cast c) {
 switch(c) {
  case Cast::adaptavg1d:      return adapt(nn::AdaptiveAvgPool1dOptions(3));
  case Cast::adaptavg2d:      return adapt(nn::AdaptiveAvgPool2dOptions({3,2}));
  case Cast::adaptavg3d:      return adapt(nn::AdaptiveAvgPool3dOptions({3,2,4}));
  case Cast::adaptmax1d:      return adapt(nn::AdaptiveMaxPool1dOptions(3));
  case Cast::adaptmax2d:      return adapt(nn::AdaptiveMaxPool2dOptions({3,2}));
  case Cast::adaptmax3d:      return adapt(nn::AdaptiveMaxPool3dOptions({3,2,4}));
  case Cast::adrop:           return drop(true,nn::AlphaDropoutOptions());
  case Cast::attention:       return attention(true,nn::MultiheadAttentionOptions(2048,8));
  case Cast::avgpool1d:       return avgpool(true,nn::AvgPool1dOptions(3));
  case Cast::avgpool2d:       return avgpool(true,nn::AvgPool2dOptions({3,2}));
  case Cast::avgpool3d:       return avgpool(true,nn::AvgPool3dOptions({3,2,2}));
  case Cast::base:            return KDICT;
  case Cast::batchnorm1d:
  case Cast::batchnorm2d:
  case Cast::batchnorm3d:     return batchnorm(true,nn::BatchNormOptions(32));
  case Cast::bilinear:        return bilinear(true,nn::BilinearOptions(20,30,40));
  case Cast::cat:             return dim(true,c,CatOptions().dim());
  case Cast::celu:            return alpha(true,nn::CELUOptions());
  case Cast::conv1d:          return conv(true,nn::detail::ConvNdOptions<1>(16,32,3));
  case Cast::conv2d:          return conv(true,nn::detail::ConvNdOptions<2>(16,32,{3,5}));
  case Cast::conv3d:          return conv(true,nn::detail::ConvNdOptions<3>(16,32,{3,5,2}));
  case Cast::convtranspose1d: return conv(true,nn::detail::ConvNdOptions<1>(128,64,5).transposed(true));
  case Cast::convtranspose2d: return conv(true,nn::detail::ConvNdOptions<2>(128,64,{3,5}).transposed(true));
  case Cast::convtranspose3d: return conv(true,nn::detail::ConvNdOptions<3>(128,64,{3,5,2}).transposed(true));
  case Cast::crossmap2d:      return localnorm(true,c,nn::CrossMapLRN2dOptions(2));
  case Cast::decoder:         return decoder(true,nn::TransformerDecoderOptions(
                                             nn::TransformerDecoderLayerOptions(512,8),6)
                                             .norm(AnyModule(nn::LayerNorm(nn::LayerNormOptions({512})))));
  case Cast::decoderlayer:    return codelayer(true,nn::TransformerDecoderLayerOptions(512,8));
  case Cast::drop:            return drop(true,nn::DropoutOptions());
  case Cast::drop2d:          return drop(true,nn::Dropout2dOptions());
  case Cast::drop3d:          return drop(true,nn::Dropout3dOptions());
  case Cast::elu:             return alpha(true,nn::ELUOptions());
  case Cast::embed:           return embed(true,c,nn::EmbeddingOptions(1000,64),{});
  case Cast::embedbag:        return embed(true,c,nn::EmbeddingBagOptions(1000,64),{});
  case Cast::encoder:         return encoder(true,nn::TransformerEncoderOptions(
                                             nn::TransformerEncoderLayerOptions(512,8),6)
                                             .norm(AnyModule(nn::LayerNorm(nn::LayerNormOptions({512})))));
  case Cast::encoderlayer:    return codelayer(true,nn::TransformerEncoderLayerOptions(512,8));
  case Cast::expand:          return getsize(true,SizeOptions({-1,-1,28,28}));
  case Cast::fadrop:          return drop(true,nn::FeatureAlphaDropoutOptions());
  case Cast::flatten:         return flatten(true,nn::FlattenOptions());
  case Cast::fmaxpool2d:      return fpool(true,nn::FractionalMaxPool2dOptions({2,4})  .output_size(ExpandingArray<2>({16,32})));
  case Cast::fmaxpool3d:      return fpool(true,nn::FractionalMaxPool3dOptions({2,4,3}).output_size(ExpandingArray<3>({16,32,24})));
  case Cast::fold:            return fold(true,nn::FoldOptions({4,6},{2,3}));
  case Cast::fork:            return KDICT;
  case Cast::gelu:            return KDICT;
  case Cast::glu:             return dim(true,c,nn::GLUOptions().dim());
  case Cast::groupnorm:       return groupnorm(true,nn::GroupNormOptions(3,6));
  case Cast::gru:             return rnn(true,nn::GRUOptions(10,20));
  case Cast::gruout:          return KDICT;
  case Cast::hardshrink:      return lambda(true,c,torch::nn::HardshrinkOptions().lambda());
  case Cast::hardtanh:        return hardtanh(true,nn::HardtanhOptions());
  case Cast::identity:        return KDICT;
  case Cast::indexselect:     return indexselect(true,IndexSelectOptions(1,torch::arange(3)));
  case Cast::instancenorm1d:
  case Cast::instancenorm2d:
  case Cast::instancenorm3d:  return batchnorm(true,nn::InstanceNormOptions(100));
  case Cast::interpolate:     return interpolate(true,fnn::InterpolateFuncOptions().size(std::vector<int64_t>({4})));
  case Cast::layernorm:       return layernorm(true,nn::LayerNormOptions({32,10}));
  case Cast::leakyrelu:       return slope(true,c,nn::LeakyReLUOptions());
  case Cast::linear:          return linear(true,nn::LinearOptions(784,10));
  case Cast::localnorm:       return localnorm(true,c,nn::LocalResponseNormOptions(2));
  case Cast::logsigmoid:      return KDICT;
  case Cast::logsoftmax:      return dim(true,c,nn::LogSoftmaxOptions(1).dim());
  case Cast::lppool1d:        return lppool(true,nn::LPPool1dOptions(2,3));
  case Cast::lppool2d:        return lppool(true,nn::LPPool2dOptions(1.2,{2,3}));
  case Cast::lstm:            return rnn(true,nn::LSTMOptions(10,20));
  case Cast::lstmout:         return KDICT;
  case Cast::maxpool1d:       return maxpool(true,nn::MaxPool1dOptions(3));
  case Cast::maxpool2d:       return maxpool(true,nn::MaxPool2dOptions({3,2}));
  case Cast::maxpool3d:       return maxpool(true,nn::MaxPool3dOptions({3,2,2}));
  case Cast::mish:            return KDICT;
  case Cast::moduledict:      return KDICT;
  case Cast::modulelist:      return KDICT;
  case Cast::mul:             return KDICT;
  case Cast::nbeats:          return KDICT;
  case Cast::normalize:       return normalize(true,fnn::NormalizeFuncOptions());
  case Cast::onehot:          return onehot(true,OneHotOptions(10));
  case Cast::pad:             return pad(true,fnn::PadFuncOptions({1, 2, 2, 1, 1, 2}));
  case Cast::pad1d:           return cpad(nn::ConstantPad1dOptions({1,2},0));
  case Cast::pad2d:           return cpad(nn::ConstantPad2dOptions({1,1,2,2},0));
  case Cast::pad3d:           return cpad(nn::ConstantPad3dOptions({3,3,6,6,0,1}, 3.5));
  case Cast::parmdict:        return KDICT;
  case Cast::pairwise:        return pairwise(true,nn::PairwiseDistanceOptions());
  case Cast::prelu:           return prelu(true,nn::PReLUOptions());
  case Cast::randomcrop:      return rcrop(true,RandomCropOptions(32,4).padmode(torch::kReflect));
  case Cast::randomflip:      return rflip(true,RandomFlipOptions(.5, -1));
  case Cast::recur:           return recur(true,RecurOptions());
  case Cast::reflect1d:       return npad(nn::ReflectionPad1dOptions({1,2}));
  case Cast::reflect2d:       return npad(nn::ReflectionPad2dOptions({1,1,2,0}));
  case Cast::relu:            return inplace(true,nn::ReLUOptions().inplace());
  case Cast::relu6:           return inplace(true,nn::ReLU6Options().inplace());
  case Cast::replicate1d:     return npad(nn::ReplicationPad1dOptions({1,2}));
  case Cast::replicate2d:     return npad(nn::ReplicationPad2dOptions({1,1,2,0}));
  case Cast::replicate3d:     return npad(nn::ReplicationPad3dOptions({3,3,6,6,1,1}));
  case Cast::residual:        return KDICT;
  case Cast::reshape:         return getsize(true,SizeOptions({-1,1,28,28}));
  case Cast::rnn:             return rnn(true,nn::RNNOptions(10,20));
  case Cast::rnnout:          return KDICT;
  case Cast::rrelu:           return rrelu(true,nn::RReLUOptions());
  case Cast::select:          return select(true,SelectOptions(1,-1));
  case Cast::selu:            return inplace(true,nn::SELUOptions().inplace());
  case Cast::seqjoin:
  case Cast::seqnest:
  case Cast::sequential:      return KDICT;
  case Cast::sigmoid:         return KDICT;
  case Cast::silu:            return KDICT;
  case Cast::similar:         return similar(true,nn::CosineSimilarityOptions());
  case Cast::softmax:         return dim(true,c,nn::SoftmaxOptions(1).dim());
  case Cast::softmax2d:       return KDICT;
  case Cast::softmin:         return dim(true,c,nn::SoftminOptions(1).dim());
  case Cast::softplus:        return softplus(true,nn::SoftplusOptions());
  case Cast::softshrink:      return lambda(true,c,nn::SoftshrinkOptions().lambda());
  case Cast::softsign:        return KDICT;
  case Cast::squeeze:         return squeeze(true,SqueezeOptions(1));
  case Cast::tanh:            return KDICT;
  case Cast::tanhshrink:      return KDICT;
  case Cast::threshold:       return threshold(true,nn::ThresholdOptions(.1,0));
  case Cast::transform:       return KDICT;
  case Cast::transformer:     return transformer(true,nn::TransformerOptions());
  case Cast::unfold:          return unfold(true,nn::UnfoldOptions({2,3}));
  case Cast::unsqueeze:       return squeeze(true,SqueezeOptions(0));
  case Cast::upsample:        return upsample(true,nn::UpsampleOptions());
  case Cast::zeropad2d:       return npad(nn::ZeroPad2dOptions({1,1,2,0}));
  case Cast::zscore:          return zscore(true,ZscoreOptions(torch::tensor({.51,.49,.47}).to(torch::kDouble),
                                                               torch::tensor({.25,.25,.21}).to(torch::kDouble)));
  case Cast::undefined: {
   const auto& e=env().module; J i=0,n=e.size();
   K k=ktn(KS,3),s=ktn(KS,n),d=ktn(0,n),o=ktn(0,n);
   kS(k)[0]=cs("module"); kS(k)[1]=cs("pytorch"); kS(k)[2]=cs("options");
   for(const auto& a:e) {
    kS(s)[i]=std::get<0>(a);
    kK(d)[i]=kp((S)std::get<3>(a).c_str());
    kK(o)[i]=modulehelp(std::get<1>(a)); ++i;
   }
   return xT(xD(k,knk(3,s,d,o)));
  }
  default: TORCH_ERROR("no help implemented for module: ",msym(c));
 }
}

// ----------------------------------------
// contain: shallow copy a container module
// ----------------------------------------
KAPI contain(K x) {
 KTRY
  Kmodule *k=xmodule(x);
  TORCH_CHECK(k, "contain: expects a module");
  auto z=mcreate(nullptr,0,k->c);
  for(const auto& c:k->m->children())
   addmodule(z,c);
  return kmodule(k->c,z,k->a);
 KCATCH("contain");
}

// ----------------------------------
// module fns defined in k namespace
// ----------------------------------
void nnfn(K x) {
 fn(x, "seq",         KFN(seq),          1);    // convenience fn for sequential layers
 fn(x, "module",      KFN(module),       1);    // api function for module create/query

 fn(x, "adaptavg1d",  KFN(adaptavg1d),   1);    // functional form of modules/activations
 fn(x, "adaptavg2d",  KFN(adaptavg2d),   1);
 fn(x, "adaptavg3d",  KFN(adaptavg3d),   1);
 fn(x, "adaptmax1d",  KFN(adaptmax1d),   1);
 fn(x, "adaptmax2d",  KFN(adaptmax2d),   1);
 fn(x, "adaptmax3d",  KFN(adaptmax3d),   1);
 fn(x, "fmaxpool2d",  KFN(fmaxpool2d),   1);
 fn(x, "fmaxpool3d",  KFN(fmaxpool3d),   1);
 fn(x, "avgpool1d",   KFN(avgpool1d),    1);
 fn(x, "avgpool2d",   KFN(avgpool2d),    1);
 fn(x, "avgpool3d",   KFN(avgpool3d),    1);
 fn(x, "pad",         KFN(kpad),         1);
 fn(x, "celu",        KFN(celu),         1);
 fn(x, "elu",         KFN(elu),          1);
 fn(x, "flatten",     KFN(Flatten),      1);
 fn(x, "fold",        KFN(Fold),         1);
 fn(x, "glu",         KFN(glu),          1);
 fn(x, "hardshrink",  KFN(hardshrink),   1);
 fn(x, "hardtanh",    KFN(Hardtanh),     1);
 fn(x, "interpolate", KFN(kinterpolate), 1);
 fn(x, "leakyrelu",   KFN(leakyrelu),    1);
 fn(x, "linear",      KFN(Linear),       1);
 fn(x, "bilinear",    KFN(Bilinear),     1);
 fn(x, "logsigmoid",  KFN(logsigmoid),   1);
 fn(x, "logsoftmax",  KFN(logsoftmax),   1);
 fn(x, "lppool1d",    KFN(lppool1d),     1);
 fn(x, "lppool2d",    KFN(lppool2d),     1);
 fn(x, "maxpool1d",   KFN(maxpool1d),    1);
 fn(x, "maxpool2d",   KFN(maxpool2d),    1);
 fn(x, "maxpool3d",   KFN(maxpool3d),    1);
 fn(x, "normalize",   KFN(Normalize),    1);
 fn(x, "onehot",      KFN(Onehot),       1);
 fn(x, "prelu",       KFN(Prelu),        1);
 fn(x, "randomcrop",  KFN(randomcrop),   1);
 fn(x, "randomflip",  KFN(randomflip),   1);
 fn(x, "relu",        KFN(relu),         1);
 fn(x, "relu6",       KFN(relu6),        1);
 fn(x, "rrelu",       KFN(Rrelu),        1);
 fn(x, "selu",        KFN(selu),         1);
 fn(x, "softmax",     KFN(softmax),      1);
 fn(x, "softmin",     KFN(softmin),      1);
 fn(x, "softplus",    KFN(Softplus),     1);
 fn(x, "softsign",    KFN(softsign),     1);
 fn(x, "softshrink",  KFN(softshrink),   1);
 fn(x, "tanhshrink",  KFN(tanhshrink),   1);
 fn(x, "threshold",   KFN(Threshold),    1);
 fn(x, "unfold",      KFN(Unfold),       1);
 fn(x, "pairwise",    KFN(Pairwise),     1);
 fn(x, "pdist",       KFN(pdist),        1);
 fn(x, "similar",     KFN(Similar),      1);
 fn(x, "zscore",      KFN(kzscore),      1);
}

/*
AdaptiveLogSoftmaxWithLoss - alternative to softmax when distribution is highly imbalanced, e.g. in language processing
normalize, interpolate  -- functional form implemented, add module?
pairwise distance & cosine similarity: in both module & functional form but forward method needs 2 input tensors
fractional pool -- try with indices registered as buffer?
embeddingbag -- forward w'defaults should work with sequential
1.7 adds UnFlatten
BaseModule - push_back ? / forward ?

GRU,RNN
 std::tuple<Tensor, Tensor> forward(const Tensor& input, Tensor hx = {});
LSTM
 std::tuple<Tensor, std::tuple<Tensor, Tensor>> forward(const Tensor& input, torch::optional<std::tuple<Tensor, Tensor>> hx_opt = {});

RNNCell, GRUCell
  Tensor forward(const Tensor& input, Tensor hx = {});
LSTMCell
  std::tuple<Tensor, Tensor> forward(const Tensor& input, torch::optional<std::tuple<Tensor, Tensor>> hx_opt = {});
*/

/* adding a new module
	- add to knn.h if not defined in pytorch
	- add Cast enumeration and entry in Env.module, may need to add to Setting's
        - if container, need to amend container functions
          also, need to modify addmodule to handle particular case..
        - if forward() result not a tensor, need to define result type and handle in forward calcs
          define forward calc for requisite input arg (usually a single tensor)
	- fns to process options, e.g. options(K x,J i,Cast c) & options(bool b,const Options& o)
	- mcreate & anymodule creation or special parent creation
	- mopt to return dictionary of options
	- modulehelp entry
*/
