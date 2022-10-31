#include "ktorch.h"
#include "knn.h"

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

std::string mlabel(const char *c) {
 auto s=c10::demangle(c);
 if(!s.find("struct "))     s.erase(s.begin(),s.begin()+7);
 if(!s.find("class "))      s.erase(s.begin(),s.begin()+6);
 if(!s.find("torch::nn::")) s.erase(s.begin(),s.begin()+11);
 if(s.find("Impl",s.size()-4)==s.size()-4) s.erase(s.size()-4,s.size());
 return s;
}

std::string mlabel(const Module& x) {
 return mlabel(typeid(x).name());
}

std::string mlabel(const Moduleptr& x) {return mlabel(*x);}
std::string mlabel(Kmodule* x) {return mlabel(x->m);}

// ------------------------------------------------------------------
// argstart - return offset in k list to begin processing module args
// ------------------------------------------------------------------
J argstart(K x,S s) {return !x ? -1 : (xdict(x) ? 0 : (s ? 2 : 1));}

// ----------------------------------------------------------------------------
// msym - map to/from sym & enum for module, e.g. `conv3d <-> Cast::conv3d
// msyms - parse module & optional name from k arg(s), throw error if not found
// ----------------------------------------------------------------------------
S msym(Cast c) {
 for(auto& m:env().modules) if(c==std::get<1>(m)) return std::get<0>(m);
 TORCH_ERROR("unrecognized module: cannot translate enumeration ",(I)c," to symbol");
}

Cast msym(S s) {
 for(const auto& m:env().modules) if(s==std::get<0>(m)) return std::get<1>(m);
 TORCH_ERROR("unrecognized module type: `",s);
}

void msyms(K x,S& s,S& nm) {
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

// -----------------------------------------------------------------------------------
// mkeys - keys for dict/table of module state: `depth`module`name`options`parms`buffers
// -----------------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// mcast - given generic module/ptr, return api enumeration, e.g. Cast::linear
// mcast - given module ptr and flag, true returns first else last child
// msym  - given generic module, return api symbol, e.g. `linear
// ---------------------------------------------------------------------------
static Cast mcast(size_t h) {
 for(const auto& m:env().modules)
  if(std::get<2>(m)==h) return std::get<1>(m);
 return Cast::undefined;
}

Cast mcast(const Module& m) {
 return m.as<knn::Callback>() ? Cast::callback : mcast(typeid(m).hash_code());
}

Cast mcast(const Moduleptr& m) {return mcast(*m);}

Cast mcast(const Moduleptr& m,bool b) {
 const auto& v=m->children();
 return v.size() ? mcast(b ? v.front() : v.back(),b) : mcast(*m);
}

static S msym(size_t h) {
 for(const auto& m:env().modules)
  if(std::get<2>(m)==h) return std::get<0>(m);
 return nullsym();
}

S msym(const Module& m) {return msym(typeid(m).hash_code());}

// ---------------------------------------------------------------------------
// findtensor - given module & parm/buffer name, return tensor pointer or null
// --------------------------------------------------------------------------
static const Tensor *findtensor(const Module& m,const std::string& s,bool p) {
 if(p) {
   if(const auto *t=access_private::parameters_(m).find(s))
    return t;
 } else {
   if(const auto *t=access_private::buffers_(m).find(s))
    return t;
 }
 auto n=s.find_first_of('.');
 if(n==std::string::npos)
  return nullptr;
 for(const auto& a:access_private::children_(m)) {
  if(!s.rfind(a.key(),0) && n==a.key().size()) {
   if(const auto* t=findtensor(*a.value(), s.substr(1+n), p))
    return t;
  }
 }
 return nullptr;
}

const Tensor *findtensor(const Module& m,const std::string& s,Cast c) {
 const Tensor *t;
 switch(c) {
  case Cast::parameter: t=findtensor(m,s,true); break;
  case Cast::buffer:    t=findtensor(m,s,false); break;
  case Cast::tensor:    if(!(t=findtensor(m,s,true))) t=findtensor(m,s,false); break;
  default: TORCH_ERROR("invalid tensor type, expecting tensor, parameter, or buffer");
 }
 return t;
}

// -------------------------------------------------------------------------------
// hasforward - return true if module has a non-templatized forward method defined
// forwardfind - look in module children for first/last module w'forward calc
// forwardoptions - define result,args and other properies of forward() function
// -------------------------------------------------------------------------------
static bool hasforward(Cast c) {
 switch(c) {
  case Cast::sequential:
  case Cast::callback:
  case Cast::moduledict:
  case Cast::modulelist:
  case Cast::parmdict:
   return false;
  default:
   return true;
 }
}

static void forwardfind(bool b,ForwardOptions& f,const Module& m) {
 auto h=typeid(m).hash_code(); // run-time type of module
 for(const auto& a:env().modules) {
  if(std::get<2>(a)==h) {
   if(std::get<4>(a)) {              // if module has non-templatized forward
    if(b) {                          // module in sequence that recieves input
     f.in(std::get<1>(a));           // get enumeration of input module type
     f.n(std::get<6>(a));            // get minimum required number of arguments
     f.m(std::get<7>(a));            // get maximum required number of arguments
     f.a(std::move(std::get<8>(a))); // get argument type(s)
    } else {
     f.f(true);                      // module in sequence that returns output
     f.out(std::get<1>(a));          // get enumeration of output model type
     f.r(std::get<5>(a));            // get result type
    }
    return;
   } else {
    auto const& x=access_private::children_(m);
    if(x.size())
     forwardfind(b, f, *(b ? x.front() : x.back()).value());
    return;
   }
  }
 }
 TORCH_ERROR("unable to determine forward calculation for ",mlabel(m));
}

void forwardoptions(Cast c,ForwardOptions& f,const Module& m) {
  if(hasforward(c)) {    // module has non-templatized forward call,
   for(const auto& a:env().modules)
    if(std::get<1>(a)==c) {
     f.in(c);
     f.out(c);
     TORCH_CHECK(std::get<4>(a), msym(c),": mismatch in module attributes, forward flag");
     f.f(true);
     f.r(std::get<5>(a));
     f.n(std::get<6>(a));
     f.m(std::get<7>(a));
     f.a(std::move(std::get<8>(a)));
    }
  } else if(c==Cast::callback) { // callback has templatized forward, but more information
   const auto& o=m.as<knn::Callback>()->options;
   f.in(c);
   f.out(c);
   f.f(true);
   f.r(o.out());
   f.a(o.in());
   f.n(f.a().size());
  } else {
   auto const& x=access_private::children_(m);
   if(x.size()) {
    forwardfind(true,  f, *x.front().value());
    forwardfind(false, f, *x.back().value());
   }
  }
}

// --------------------------------------------------------------------
// callbacks - return a list of callbacks based on result type & arg(s)
// fattr - placeholder function for functional calls without modules
// fwdattr - if non-templatized forward() derive result & arg types
// attrs - get attributes for modules
// moduleattrs - return list of all module attributes (for global env)
// --------------------------------------------------------------------
Callbacks callbacks() {return knn::callbacks();}

static Attrs fattr(size_t n,S s,Cast c,const char*d,const std::type_info& t) {
 return std::make_tuple(s, c, t.hash_code(), d, true, Arg::tensor, n, n, Args{Arg::tensor});
}

template<class M,class R,class... A> static Attrs fwdattr(size_t n,S s,Cast c,const char *d,size_t h,R (M::*)(A...)) {
 auto m=sizeof...(A);
 static_assert(torch::detail::check_not_lvalue_references<A...>(), "module arg(s) must not take non-const references (use pointers)");
 static_assert(!std::is_void<R>::value, "module forward calculation returns void (use dummy arg)");
 if(!n) n=m;
 return std::make_tuple(s, c, h, d, true, knn::argmap<R>(), n, m, knn::argvector<A...>());
}

template <class M,typename std::enable_if_t<!torch::detail::has_forward<M>::value>* = nullptr>
static Attrs attrs(size_t n,S s,Cast c,const char*d) {
 return std::make_tuple(s, c, typeid(M).hash_code(), d, false, Arg::tensor, n, n, Args{Arg::tensor});
}
 
template <class M,typename std::enable_if_t<torch::detail::has_forward<M>::value>* = nullptr>
static Attrs attrs(size_t n,S s,Cast c,const char*d) {
 return fwdattr(n, s, c, d, typeid(M).hash_code(), &M::forward);
}

// non-tensor arguments to forward() call:
// MultiHeadAttention uses a boolean flag
// ConvTranspose[1-3]d uses IntArrayRef for output size
// LSTM uses optional tuple with hidden state
// MaxPool and AdaptiveMaxPool have std::tuple<Tensor, Tensor> forward_with_indices(const Tensor& input);
// MaxUnpool has  Tensor forward(const Tensor& input, const Tensor& indices, const c10::optional<std::vector<int64_t>>& output_size = c10::nullopt);

ModuleAttrs moduleattrs() {
 return {{
  attrs<nn::AdaptiveAvgPool1dImpl>      (0, cs("adaptavg1d"),      Cast::adaptavg1d,       "torch.nn.AdaptiveAvgPool1d"),
  attrs<nn::AdaptiveAvgPool2dImpl>      (0, cs("adaptavg2d"),      Cast::adaptavg2d,       "torch.nn.AdaptiveAvgPool2d"),
  attrs<nn::AdaptiveAvgPool3dImpl>      (0, cs("adaptavg3d"),      Cast::adaptavg3d,       "torch.nn.AdaptiveAvgPool3d"),
  attrs<nn::AdaptiveMaxPool1dImpl>      (0, cs("adaptmax1d"),      Cast::adaptmax1d,       "torch.nn.AdaptiveMaxPool1d"),
  attrs<nn::AdaptiveMaxPool2dImpl>      (0, cs("adaptmax2d"),      Cast::adaptmax2d,       "torch.nn.AdaptiveMaxPool2d"),
  attrs<nn::AdaptiveMaxPool3dImpl>      (0, cs("adaptmax3d"),      Cast::adaptmax3d,       "torch.nn.AdaptiveMaxPool3d"),
  attrs<nn::AlphaDropoutImpl>           (0, cs("adrop"),           Cast::adrop,            "torch.nn.AlphaDropout"),
  attrs<nn::MultiheadAttentionImpl>     (3, cs("attention"),       Cast::attention,        "torch.nn.MultiheadAttention"),
  attrs<nn::AvgPool1dImpl>              (0, cs("avgpool1d"),       Cast::avgpool1d,        "torch.nn.AvgPool1d"),
  attrs<nn::AvgPool2dImpl>              (0, cs("avgpool2d"),       Cast::avgpool2d,        "torch.nn.AvgPool2d"),
  attrs<nn::AvgPool3dImpl>              (0, cs("avgpool3d"),       Cast::avgpool3d,        "torch.nn.AvgPool3d"),
  attrs<nn::BatchNorm1dImpl>            (0, cs("batchnorm1d"),     Cast::batchnorm1d,      "torch.nn.BatchNorm1d"),
  attrs<nn::BatchNorm2dImpl>            (0, cs("batchnorm2d"),     Cast::batchnorm2d,      "torch.nn.BatchNorm2d"),
  attrs<nn::BatchNorm3dImpl>            (0, cs("batchnorm3d"),     Cast::batchnorm3d,      "torch.nn.BatchNorm3d"),
  attrs<nn::BilinearImpl>               (0, cs("bilinear"),        Cast::bilinear,         "torch.nn.Bilinear"),
  attrs<knn::CallbackImpl>              (1, cs("callback"),        Cast::callback,         "knn.Callback"),
  attrs<knn::CatImpl>                   (0, cs("cat"),             Cast::cat,              "torch.cat"),
  attrs<nn::CELUImpl>                   (0, cs("celu"),            Cast::celu,             "torch.nn.CELU"),
  attrs<nn::Conv1dImpl>                 (0, cs("conv1d"),          Cast::conv1d,           "torch.nn.Conv1d"),
  attrs<nn::Conv2dImpl>                 (0, cs("conv2d"),          Cast::conv2d,           "torch.nn.Conv2d"),
  attrs<nn::Conv3dImpl>                 (0, cs("conv3d"),          Cast::conv3d,           "torch.nn.Conv3d"),
  attrs<nn::ConvTranspose1dImpl>        (1, cs("convtranspose1d"), Cast::convtranspose1d,  "torch.nn.ConvTranspose1d"),
  attrs<nn::ConvTranspose2dImpl>        (1, cs("convtranspose2d"), Cast::convtranspose2d,  "torch.nn.ConvTranspose2d"),
  attrs<nn::ConvTranspose3dImpl>        (1, cs("convtranspose3d"), Cast::convtranspose3d,  "torch.nn.ConvTranspose3d"),
  attrs<nn::CrossMapLRN2dImpl>          (0, cs("crossmap2d"),      Cast::crossmap2d,       "torch.nn.CrossMapLRN2d"),
  attrs<nn::TransformerDecoderImpl>     (2, cs("decoder"),         Cast::decoder,          "torch.nn.TransformerDecoder"),
  attrs<nn::TransformerDecoderLayerImpl>(2, cs("decoderlayer"),    Cast::decoderlayer,     "torch.nn.TransformerDecoderLayer"),
  attrs<nn::DropoutImpl>                (0, cs("drop"),            Cast::drop,             "torch.nn.Dropout"),
  attrs<nn::Dropout2dImpl>              (0, cs("drop2d"),          Cast::drop2d,           "torch.nn.Dropout2d"),
  attrs<nn::Dropout3dImpl>              (0, cs("drop3d"),          Cast::drop3d,           "torch.nn.Dropout3d"),
  attrs<knn::DropPathImpl>              (0, cs("droppath"),        Cast::droppath,         "knn.DropPath"),
  attrs<nn::ELUImpl>                    (0, cs("elu"),             Cast::elu,              "torch.nn.ELU"),
  attrs<nn::EmbeddingImpl>              (0, cs("embed"),           Cast::embed,            "torch.nn.Embedding"),
  attrs<nn::EmbeddingBagImpl>           (1, cs("embedbag"),        Cast::embedbag,         "torch.nn.EmbeddingBag"),
  attrs<knn::EmbedPositionImpl>         (0, cs("embedpos"),        Cast::embedpos,         "knn.EmbedPosition"),
  attrs<knn::EmbedSequenceImpl>         (0, cs("embedseq"),        Cast::embedseq,         "knn.EmbedSequence"),
  attrs<nn::TransformerEncoderImpl>     (1, cs("encoder"),         Cast::encoder,          "torch.nn.TransformerEncoder"),
  attrs<nn::TransformerEncoderLayerImpl>(1, cs("encoderlayer"),    Cast::encoderlayer,     "torch.nn.TransformerEncoderLayer"),
  attrs<knn::ExpandImpl>                (0, cs("expand"),          Cast::expand,           "torch.Tensor.expand"),
  attrs<nn::FeatureAlphaDropoutImpl>    (0, cs("fadrop"),          Cast::fadrop,           "torch.nn.FeatureAlphaDropout"),
  attrs<nn::FlattenImpl>                (0, cs("flatten"),         Cast::flatten,          "torch.nn.Flatten"),
  attrs<nn::FractionalMaxPool2dImpl>    (0, cs("fmaxpool2d"),      Cast::fmaxpool2d,       "torch.nn.FractionalMaxPool2d"),
  attrs<nn::FractionalMaxPool3dImpl>    (0, cs("fmaxpool3d"),      Cast::fmaxpool3d,       "torch.nn.FractionalMaxPool3d"),
  attrs<nn::FoldImpl>                   (0, cs("fold"),            Cast::fold,             "torch.nn.Fold"),
  attrs<knn::ForkImpl>                  (0, cs("fork"),            Cast::fork,             "knn.Fork"),
  attrs<nn::GELUImpl>                   (0, cs("gelu"),            Cast::gelu,             "torch.nn.GELU"),
  attrs<nn::GLUImpl>                    (0, cs("glu"),             Cast::glu,              "torch.nn.GLU"),
  attrs<nn::GroupNormImpl>              (0, cs("groupnorm"),       Cast::groupnorm,        "torch.nn.GroupNorm"),
  attrs<nn::GRUImpl>                    (1, cs("gru"),             Cast::gru,              "torch.nn.GRU"),
  attrs<nn::HardshrinkImpl>             (0, cs("hardshrink"),      Cast::hardshrink,       "torch.nn.Hardshrink"),
  attrs<nn::HardtanhImpl>               (0, cs("hardtanh"),        Cast::hardtanh,         "torch.nn.Hardtanh"),
  attrs<nn::IdentityImpl>               (0, cs("identity"),        Cast::identity,         "torch.nn.Identity"),
  attrs<knn::IndexSelectImpl>           (0, cs("indexselect"),     Cast::indexselect,      "torch.index_select"),
  attrs<nn::InstanceNorm1dImpl>         (0, cs("instancenorm1d"),  Cast::instancenorm1d,   "torch.nn.InstanceNorm1d"),
  attrs<nn::InstanceNorm2dImpl>         (0, cs("instancenorm2d"),  Cast::instancenorm2d,   "torch.nn.InstanceNorm2d"),
  attrs<nn::InstanceNorm3dImpl>         (0, cs("instancenorm3d"),  Cast::instancenorm3d,   "torch.nn.InstanceNorm3d"),
  fattr                                 (1, cs("interpolate"),     Cast::interpolate,      "torch.nn.functional.interpolate", typeid(fnn::interpolate)),
  attrs<nn::LayerNormImpl>              (0, cs("layernorm"),       Cast::layernorm,        "torch.nn.LayerNorm"),
  attrs<nn::LeakyReLUImpl>              (0, cs("leakyrelu"),       Cast::leakyrelu,        "torch.nn.LeakyReLU"),
  attrs<nn::LinearImpl>                 (0, cs("linear"),          Cast::linear,           "torch.nn.Linear"),
  attrs<nn::LocalResponseNormImpl>      (0, cs("localnorm"),       Cast::localnorm,        "torch.nn.LocalResponseNorm"),
  attrs<nn::LogSigmoidImpl>             (0, cs("logsigmoid"),      Cast::logsigmoid,       "torch.nn.LogSigmoid"),
  attrs<nn::LogSoftmaxImpl>             (0, cs("logsoftmax"),      Cast::logsoftmax,       "torch.nn.LogSoftmax"),
  attrs<nn::LPPool1dImpl>               (0, cs("lppool1d"),        Cast::lppool1d,         "torch.nn.LPPool1d"),
  attrs<nn::LPPool2dImpl>               (0, cs("lppool2d"),        Cast::lppool2d,         "torch.nn.LPPool2d"),
  attrs<nn::LSTMImpl>                   (1, cs("lstm"),            Cast::lstm,             "torch.nn.LSTM"),
  attrs<knn::MatmulImpl>                (0, cs("matmul"),          Cast::matmul,           "torch.matmul"),
  attrs<nn::MaxPool1dImpl>              (0, cs("maxpool1d"),       Cast::maxpool1d,        "torch.nn.MaxPool1d"),
  attrs<nn::MaxPool2dImpl>              (0, cs("maxpool2d"),       Cast::maxpool2d,        "torch.nn.MaxPool2d"),
  attrs<nn::MaxPool3dImpl>              (0, cs("maxpool3d"),       Cast::maxpool3d,        "torch.nn.MaxPool3d"),
  attrs<nn::MishImpl>                   (0, cs("mish"),            Cast::mish,             "torch.nn.Mish"),
  attrs<nn::ModuleDictImpl>             (0, cs("moduledict"),      Cast::moduledict,       "torch.nn.ModuleDict"),
  attrs<nn::ModuleListImpl>             (0, cs("modulelist"),      Cast::modulelist,       "torch.nn.ModuleList"),
  attrs<knn::MulImpl>                   (0, cs("mul"),             Cast::mul,              "torch.mul"),
  attrs<knn::NBeatsImpl>                (0, cs("nbeats"),          Cast::nbeats,           "knn.NBeats"),
  fattr                                 (1, cs("normalize"),       Cast::normalize,        "torch.nn.functional.normalize", typeid(fnn::normalize)),
  attrs<knn::OneHotImpl>                (0, cs("onehot"),          Cast::onehot,           "torch.nn.functional.one_hot"),
  attrs<knn::PadImpl>                   (0, cs("pad"),             Cast::pad,              "torch.nn.functional.pad"),
  attrs<nn::ConstantPad1dImpl>          (0, cs("pad1d"),           Cast::pad1d,            "torch.nn.ConstantPad1d"),
  attrs<nn::ConstantPad2dImpl>          (0, cs("pad2d"),           Cast::pad2d,            "torch.nn.ConstantPad2d"),
  attrs<nn::ConstantPad3dImpl>          (0, cs("pad3d"),           Cast::pad3d,            "torch.nn.ConstantPad3d"),
  attrs<nn::PairwiseDistanceImpl>       (0, cs("pairwise"),        Cast::pairwise,         "torch.nn.PairwiseDistance"),
  attrs<nn::ParameterDictImpl>          (0, cs("parmdict"),        Cast::parmdict,         "torch.nn.ParameterDict"),
  attrs<knn::PermuteImpl>               (0, cs("permute"),         Cast::permute,          "torch.permute"),
  attrs<nn::PReLUImpl>                  (0, cs("prelu"),           Cast::prelu,            "torch.nn.PReLU"),
  attrs<knn::RandomCropImpl>            (0, cs("randomcrop"),      Cast::randomcrop,       "torchvision.transforms"),
  attrs<knn::RandomFlipImpl>            (0, cs("randomflip"),      Cast::randomflip,       "torchvision.transforms"),
  attrs<knn::RecurImpl>                 (1, cs("recur"),           Cast::recur,            "knn.Recur"),
  attrs<nn::ReflectionPad1dImpl>        (0, cs("reflect1d"),       Cast::reflect1d,        "torch.nn.ReflectionPad1d"),
  attrs<nn::ReflectionPad2dImpl>        (0, cs("reflect2d"),       Cast::reflect2d,        "torch.nn.ReflectionPad2d"),
  attrs<nn::ReLUImpl>                   (0, cs("relu"),            Cast::relu,             "torch.nn.ReLU"),
  attrs<nn::ReLU6Impl>                  (0, cs("relu6"),           Cast::relu6,            "torch.nn.ReLU6"),
  attrs<nn::ReplicationPad1dImpl>       (0, cs("replicate1d"),     Cast::replicate1d,      "torch.nn.ReplicationPad1d"),
  attrs<nn::ReplicationPad2dImpl>       (0, cs("replicate2d"),     Cast::replicate2d,      "torch.nn.ReplicationPad2d"),
  attrs<nn::ReplicationPad3dImpl>       (0, cs("replicate3d"),     Cast::replicate3d,      "torch.nn.ReplicationPad3d"),
  attrs<knn::ReshapeImpl>               (0, cs("reshape"),         Cast::reshape,          "torch.reshape"),
  attrs<knn::ResidualImpl>              (1, cs("residual"),        Cast::residual,         "knn.Residual"),
  attrs<nn::RNNImpl>                    (1, cs("rnn"),             Cast::rnn,              "torch.nn.RNN"),
  attrs<nn::RReLUImpl>                  (0, cs("rrelu"),           Cast::rrelu,            "torch.nn.RReLU"),
  attrs<knn::SelectImpl>                (0, cs("select"),          Cast::select,           "torcn.Tensor.select"),
  attrs<knn::SelfAttentionImpl>         (1, cs("selfattention"),   Cast::selfattention,    "knn.SelfAttention"),
  attrs<nn::SELUImpl>                   (0, cs("selu"),            Cast::selu,             "torch.nn.SELU"),
  attrs<knn::SeqJoinImpl>               (1, cs("seqjoin"),         Cast::seqjoin,          "knn.SeqJoin"),
  attrs<knn::SeqDictImpl>               (1, cs("seqdict"),         Cast::seqdict,          "knn.SeqDict"),
  attrs<knn::SeqListImpl>               (1, cs("seqlist"),         Cast::seqlist,          "knn.SeqList"),
  attrs<knn::SeqNestImpl>               (1, cs("seqnest"),         Cast::seqnest,          "knn.SeqNest"),
  attrs<nn::SequentialImpl>             (1, cs("sequential"),      Cast::sequential,       "torch.nn.Sequential"),
  attrs<nn::SigmoidImpl>                (0, cs("sigmoid"),         Cast::sigmoid,          "torch.nn.Sigmoid"),
  attrs<nn::SiLUImpl>                   (0, cs("silu"),            Cast::silu,             "torch.nn.SiLU"),
  attrs<nn::CosineSimilarityImpl>       (0, cs("similar"),         Cast::similar,          "torch.nn.CosineSimilarity"),
  attrs<nn::SoftmaxImpl>                (0, cs("softmax"),         Cast::softmax,          "torch.nn.Softmax"),
  attrs<nn::Softmax2dImpl>              (0, cs("softmax2d"),       Cast::softmax2d,        "torch.nn.Softmax2d"),
  attrs<nn::SoftminImpl>                (0, cs("softmin"),         Cast::softmin,          "torch.nn.Softmin"),
  attrs<nn::SoftplusImpl>               (0, cs("softplus"),        Cast::softplus,         "torch.nn.Softplus"),
  attrs<nn::SoftshrinkImpl>             (0, cs("softshrink"),      Cast::softshrink,       "torch.nn.Softshrink"),
  attrs<nn::SoftsignImpl>               (0, cs("softsign"),        Cast::softsign,         "torch.nn.Softsign"),
  attrs<knn::SqueezeImpl>               (0, cs("squeeze"),         Cast::squeeze,          "torch.squeeze"),
  attrs<nn::TanhImpl>                   (0, cs("tanh"),            Cast::tanh,             "torch.nn.Tanh"),
  attrs<nn::TanhshrinkImpl>             (0, cs("tanhshrink"),      Cast::tanhshrink,       "torch.nn.Tanhshrink"),
  attrs<nn::ThresholdImpl>              (0, cs("threshold"),       Cast::threshold,        "torch.nn.Threshold"),
  attrs<knn::TransformImpl>             (0, cs("transform"),       Cast::transform,        "knn.Transform"),
  attrs<nn::TransformerImpl>            (2, cs("transformer"),     Cast::transformer,      "torch.nn.Transformer"),
  attrs<knn::TransposeImpl>             (0, cs("transpose"),       Cast::transpose,        "knn.Transpose"),
  attrs<nn::UnfoldImpl>                 (0, cs("unfold"),          Cast::unfold,           "torch.nn.Unfold"),
  attrs<knn::UnsqueezeImpl>             (0, cs("unsqueeze"),       Cast::unsqueeze,        "torch.unsqueeze"),
  attrs<nn::UpsampleImpl>               (0, cs("upsample"),        Cast::upsample,         "torch.nn.Upsample"),
  attrs<nn::ZeroPad2dImpl>              (0, cs("zeropad2d"),       Cast::zeropad2d,        "torch.nn.ZeroPad2d"),
  attrs<knn::ZscoreImpl>                (0, cs("zscore"),          Cast::zscore,           "torchvision.transforms")
 }};
}

// ------------------------------------------------------------------------------
// kmodule - allocate object to store a module pointer (class defaults to module) 
// to - given module & options, change device/data type
// ------------------------------------------------------------------------------
K kmodule(Cast c,const Moduleptr& m,Class a) {return kptr(new Kmodule(a,c,m));}
K kmodule(Kmodule* m) {return kmodule(m->c,m->m,m->a);}

void to(Kmodule *m,const TensorOptions& o,bool a) {
 TORCH_CHECK( !(o.has_layout() || o.has_requires_grad() || o.has_pinned_memory() || o.has_memory_format()),
             "to: converts device & type, but cannot be used for layout,gradient,pinned memory or memory format");
 auto s=torch::typeMetaToScalarType(o.dtype());
 if(o.has_device() && o.has_dtype()) {
   m->module().to(o.device(),s,a);
   m->d=o.device();
 } else if(o.has_device()) {
   m->module().to(o.device(),a);
   m->d=o.device();
 } else {
  m->module().to(s,a);
 }
}

// --------------------------------------------------------------------------------------
// container - given module/module cast, return true if container module
// --------------------------------------------------------------------------------------
static bool container(Cast c) {
 switch(c) {
  case Cast::sequential:
  case Cast::seqdict:
  case Cast::seqlist:
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
  case Cast::callback:
   return true;
  default: return false;
 }
}

static bool container(const Module& m) {
 if     (m.as<nn::Sequential>())      return true;
 else if(m.as<knn::SeqNest>())        return true;
 else if(m.as<knn::SeqJoin>())        return true;
 else if(m.as<nn::ModuleDict>())      return true;
 else if(m.as<nn::ModuleList>())      return true;
 else if(m.as<nn::ParameterDict>())   return true;
 else if(m.as<knn::Fork>())           return true;
 else if(m.as<knn::NBeats>())         return true;
 else if(m.as<knn::Recur>())          return true;
 else if(m.as<knn::Residual>())       return true;
 else if(m.as<knn::Transform>())      return true;
 else if(m.as<knn::Callback>())       return true;
 else                                 return false;
}

static bool container(const Moduleptr& p) {return p ? container(*p) : false;}


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

// --------------------------------------------------------------------------------------
// parmdict - parameter dictionary handles "options" of dictionary of tensors or k arrays
// --------------------------------------------------------------------------------------
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
 if(m) {
  forwardoptions(m->c, m->f, m->module());  // update options on the forward call
  return (K)0;                              // pointer already holds the updated module
 } else {                                   // else new module being defined
  return kmodule(c,a);                      // return pointer to new module
 }
}

// -----------------------------------------------------------------------------
// tensorargs - return true if all args to forward calculation are tensors
// argstring - return string of arg(s) given vector of enumerations
// -----------------------------------------------------------------------------
static bool tensorargs(const Args& a) {
 for(auto i:a)
  if(i != Arg::tensor)
   return false;
 return a.size();
}

static std::string argstring(const Args& a) {
 std::string s;
 for(auto i:a) s += argname(i), s += ",";
 if(s.size()) s.pop_back();
 return s;
}

// -----------------------------------------------------------------------
// rforward - sequential & callback forward calc w'different return types
// -----------------------------------------------------------------------
template<typename M,typename ...X> static Output rforward(Module& m,Arg r,X... x) {
 switch(r) {
  case Arg::tensor: return m.as<M>()->template forward(x...);
  case Arg::tuple:  return m.as<M>()->template forward<Tuple>(x...);
  case Arg::nested: return m.as<M>()->template forward<Nested>(x...);
  case Arg::vector: return m.as<M>()->template forward<TensorVector>(x...);
  default: TORCH_ERROR(mlabel(m)," forward calculation returning ",argname(r)," not implemented");
 }
}

// ------------------------------------------------------------------------
// tforward - given module, run forward calc on tensor x (most common case)
// ------------------------------------------------------------------------
Output tforward(Cast c,Kmodule *k,const Tensor& x) {
 Module& m=k->module();
 switch(c) {
  case Cast::sequential:      return rforward<nn::Sequential>(m, k->f.r(), x);
  case Cast::adaptavg1d:      return m.as<nn::AdaptiveAvgPool1d>()->forward(x);
  case Cast::adaptavg2d:      return m.as<nn::AdaptiveAvgPool2d>()->forward(x);
  case Cast::adaptavg3d:      return m.as<nn::AdaptiveAvgPool3d>()->forward(x);
  case Cast::adaptmax1d:      return m.as<nn::AdaptiveMaxPool1d>()->forward(x);
  case Cast::adaptmax2d:      return m.as<nn::AdaptiveMaxPool2d>()->forward(x);
  case Cast::adaptmax3d:      return m.as<nn::AdaptiveMaxPool3d>()->forward(x);
  case Cast::adrop:           return m.as<nn::AlphaDropout>()->forward(x);
  case Cast::avgpool1d:       return m.as<nn::AvgPool1d>()->forward(x);
  case Cast::avgpool2d:       return m.as<nn::AvgPool2d>()->forward(x);
  case Cast::avgpool3d:       return m.as<nn::AvgPool3d>()->forward(x);
  case Cast::batchnorm1d:     return m.as<nn::BatchNorm1d>()->forward(x);
  case Cast::batchnorm2d:     return m.as<nn::BatchNorm2d>()->forward(x);
  case Cast::batchnorm3d:     return m.as<nn::BatchNorm3d>()->forward(x);
  case Cast::callback:        return rforward<knn::Callback>(m, k->f.r(), x);
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
  case Cast::droppath:        return m.as<knn::DropPath>()->forward(x);
  case Cast::elu:             return m.as<nn::ELU>()->forward(x);
  case Cast::embed:           return m.as<nn::Embedding>()->forward(x);
  case Cast::embedbag:        return m.as<nn::EmbeddingBag>()->forward(x);
  case Cast::embedpos:        return m.as<knn::EmbedPosition>()->forward(x);
  case Cast::embedseq:        return m.as<knn::EmbedSequence>()->forward(x);
  case Cast::encoder:         return m.as<nn::TransformerEncoder>()->forward(x);
  case Cast::encoderlayer:    return m.as<nn::TransformerEncoderLayer>()->forward(x);
  case Cast::expand:          return m.as<knn::Expand>()->forward(x);
  case Cast::fadrop:          return m.as<nn::FeatureAlphaDropout>()->forward(x);
  case Cast::flatten:         return m.as<nn::Flatten>()->forward(x);
  case Cast::fmaxpool2d:      return m.as<nn::FractionalMaxPool2d>()->forward(x);
  case Cast::fmaxpool3d:      return m.as<nn::FractionalMaxPool3d>()->forward(x);
  case Cast::fold:            return m.as<nn::Fold>()->forward(x);
  case Cast::fork:            return m.as<knn::Fork>()->forward(x);
  case Cast::gelu:            return m.as<nn::GELU>()->forward(x);
  case Cast::glu:             return m.as<nn::GLU>()->forward(x);
  case Cast::groupnorm:       return m.as<nn::GroupNorm>()->forward(x);
  case Cast::gru:             return m.as<nn::GRU>()->forward(x);
  case Cast::hardshrink:      return m.as<nn::Hardshrink>()->forward(x);
  case Cast::hardtanh:        return m.as<nn::Hardtanh>()->forward(x);
  case Cast::identity:        return m.as<nn::Identity>()->forward(x);
  case Cast::indexselect:     return m.as<knn::IndexSelect>()->forward(x);
  case Cast::instancenorm1d:  return m.as<nn::InstanceNorm1d>()->forward(x);
  case Cast::instancenorm2d:  return m.as<nn::InstanceNorm2d>()->forward(x);
  case Cast::instancenorm3d:  return m.as<nn::InstanceNorm3d>()->forward(x);
  case Cast::layernorm:       return m.as<nn::LayerNorm>()->forward(x);
  case Cast::leakyrelu:       return m.as<nn::LeakyReLU>()->forward(x);
  case Cast::linear:          return m.as<nn::Linear>()->forward(x);
  case Cast::localnorm:       return m.as<nn::LocalResponseNorm>()->forward(x);
  case Cast::logsigmoid:      return m.as<nn::LogSigmoid>()->forward(x);
  case Cast::logsoftmax:      return m.as<nn::LogSoftmax>()->forward(x);
  case Cast::lppool1d:        return m.as<nn::LPPool1d>()->forward(x);
  case Cast::lppool2d:        return m.as<nn::LPPool2d>()->forward(x);
  case Cast::lstm:            return m.as<nn::LSTM>()->forward(x);
  case Cast::maxpool1d:       return m.as<nn::MaxPool1d>()->forward(x);
  case Cast::maxpool2d:       return m.as<nn::MaxPool2d>()->forward(x);
  case Cast::maxpool3d:       return m.as<nn::MaxPool3d>()->forward(x);
  case Cast::mish:            return m.as<nn::Mish>()->forward(x);
  case Cast::nbeats:          return m.as<knn::NBeats>()->forward(x);
  case Cast::onehot:          return m.as<knn::OneHot>()->forward(x);
  case Cast::pad:             return m.as<knn::Pad>()->forward(x);
  case Cast::pad1d:           return m.as<nn::ConstantPad1d>()->forward(x);
  case Cast::pad2d:           return m.as<nn::ConstantPad2d>()->forward(x);
  case Cast::pad3d:           return m.as<nn::ConstantPad3d>()->forward(x);
  case Cast::permute:         return m.as<knn::Permute>()->forward(x);
  case Cast::prelu:           return m.as<nn::PReLU>()->forward(x);
  case Cast::randomcrop:      return m.as<knn::RandomCrop>()->forward(x);
  case Cast::randomflip:      return m.as<knn::RandomFlip>()->forward(x);
  case Cast::recur:           return m.as<knn::Recur>()->forward(x);
  case Cast::reflect1d:       return m.as<nn::ReflectionPad1d>()->forward(x);
  case Cast::reflect2d:       return m.as<nn::ReflectionPad2d>()->forward(x);
  case Cast::relu:            return m.as<nn::ReLU>()->forward(x);
  case Cast::relu6:           return m.as<nn::ReLU6>()->forward(x);
  case Cast::replicate1d:     return m.as<nn::ReplicationPad1d>()->forward(x);
  case Cast::replicate2d:     return m.as<nn::ReplicationPad2d>()->forward(x);
  case Cast::replicate3d:     return m.as<nn::ReplicationPad3d>()->forward(x);
  case Cast::residual:        return m.as<knn::Residual>()->forward(x);
  case Cast::reshape:         return m.as<knn::Reshape>()->forward(x);
  case Cast::rnn:             return m.as<nn::RNN>()->forward(x);
  case Cast::rrelu:           return m.as<nn::RReLU>()->forward(x);
  case Cast::select:          return m.as<knn::Select>()->forward(x);
  case Cast::selfattention:   return m.as<knn::SelfAttention>()->forward(x);
  case Cast::selu:            return m.as<nn::SELU>()->forward(x);
  case Cast::seqjoin:         return m.as<knn::SeqJoin>()->forward(x);
  case Cast::seqlist:         return m.as<knn::SeqList>()->forward(x);
  case Cast::seqnest:         return m.as<knn::SeqNest>()->forward(x);
  case Cast::sigmoid:         return m.as<nn::Sigmoid>()->forward(x);
  case Cast::silu:            return m.as<nn::SiLU>()->forward(x);
  case Cast::softmax:         return m.as<nn::Softmax>()->forward(x);
  case Cast::softmax2d:       return m.as<nn::Softmax2d>()->forward(x);
  case Cast::softmin:         return m.as<nn::Softmin>()->forward(x);
  case Cast::softplus:        return m.as<nn::Softplus>()->forward(x);
  case Cast::softshrink:      return m.as<nn::Softshrink>()->forward(x);
  case Cast::softsign:        return m.as<nn::Softsign>()->forward(x);
  case Cast::squeeze:         return m.as<knn::Squeeze>()->forward(x);
  case Cast::tanh:            return m.as<nn::Tanh>()->forward(x);
  case Cast::tanhshrink:      return m.as<nn::Tanhshrink>()->forward(x);
  case Cast::threshold:       return m.as<nn::Threshold>()->forward(x);
  case Cast::transform:       return m.as<knn::Transform>()->forward(x);
  case Cast::transpose:       return m.as<knn::Transpose>()->forward(x);
  case Cast::unfold:          return m.as<nn::Unfold>()->forward(x);
  case Cast::unsqueeze:       return m.as<knn::Unsqueeze>()->forward(x);
  case Cast::upsample:        return m.as<nn::Upsample>()->forward(x);
  case Cast::zeropad2d:       return m.as<nn::ZeroPad2d>()->forward(x);
  case Cast::zscore:          return m.as<knn::Zscore>()->forward(x);
  default: TORCH_ERROR("forward calculation with a single tensor argument not implemented for ",msym(c)," module");
 }
}

// ---------------------------------------------------------------------
// xforward - forward calculation for tuple, nested & tensor,tuple input
// ---------------------------------------------------------------------
static Output xforward(Cast c,Kmodule* k,const Tuple& x) {
 TORCH_ERROR(msym(k->f.in()),": forward calculation with tuple input not implemented");
}

static Output xforward(Cast c,Kmodule* k,const Nested& x) {
 TORCH_ERROR(msym(k->f.in()),": forward calculation with nested input not implemented");
}

static Output xforward(Cast c,Kmodule* k,const Tensor& x,const Tuple& y) {
 auto& m=k->module();
 switch(c) {
  case Cast::sequential: return rforward<nn::Sequential>(m, k->f.r(), x, y);
  case Cast::lstm:       return m.as<nn::LSTM>()->forward(x,y);
  default: TORCH_ERROR(msym(k->f.in()),": forward calculation with tensor,tuple input not implemented");
 }
}

// -----------------------------------------------------------------
// vforward1 - forward calc with single arg of tensor vector
// vforward3 - forward calc on 2-3 tensors from tensor vector
// vforward6 - forward calc on 2-6 tensors from tensor vector
// vforward8 - forward calc on 2-8 tensors from tensor vector
// -----------------------------------------------------------------
static Output vforward1(Cast c,Arg r,Module& m,const TensorVector& x) {
 switch(c) {
  case Cast::sequential: return rforward<nn::Sequential>(m,r,x);
  case Cast::callback:   return rforward<knn::Callback>(m,r,x);
  default: TORCH_ERROR(msym(c),": forward calculation with vector argument not implemented");
 }
}

template<typename M> static Output vforward3(Cast c,Module& m,const TensorVector& x) {
 switch(x.size()) {
  case 2: return m.as<M>()->forward(x[0], x[1]);
  case 3: return m.as<M>()->forward(x[0], x[1], x[2]);
  default: TORCH_ERROR(msym(c),": no forward calculation implemented for ",x.size()," tensor(s)");
 }
}
 
template<typename M> static Output vforward6(Cast c,Module& m,const TensorVector& x) {
 switch(x.size()) {
  case 2: return m.as<M>()->forward(x[0], x[1]);
  case 3: return m.as<M>()->forward(x[0], x[1], x[2]);
  case 4: return m.as<M>()->forward(x[0], x[1], x[2], x[3]);
  case 5: return m.as<M>()->forward(x[0], x[1], x[2], x[3], x[4]);
  case 6: return m.as<M>()->forward(x[0], x[1], x[2], x[3], x[4], x[5]);
  default: TORCH_ERROR(msym(c),": no forward calculation implemented for ",x.size()," tensor(s)");
 }
}

template<typename M> static Output vforward8(Cast c,Module& m,const TensorVector& x) {
 switch(x.size()) {
  case 2: return m.as<M>()->forward(x[0], x[1]);
  case 3: return m.as<M>()->forward(x[0], x[1], x[2]);
  case 4: return m.as<M>()->forward(x[0], x[1], x[2], x[3]);
  case 5: return m.as<M>()->forward(x[0], x[1], x[2], x[3], x[4]);
  case 6: return m.as<M>()->forward(x[0], x[1], x[2], x[3], x[4], x[5]);
  case 7: return m.as<M>()->forward(x[0], x[1], x[2], x[3], x[4], x[5], x[6]);
  case 8: return m.as<M>()->forward(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]);
  default: TORCH_ERROR(msym(c),": no forward calculation implemented for ",x.size()," tensor(s)");
 }
}
 
// -----------------------------------------------------------------
// vforward - forward calc with tensor vector w'two or more tensors
// -----------------------------------------------------------------
static Output vforward(Cast c,Kmodule *k,const TensorVector& x) {
 TORCH_CHECK(x.size()>=2, "forward calculation expects 2 or more tensors, ",x.size()," supplied");
 Module& m=k->module();
 switch(c) {
  case Cast::sequential:
    switch(x.size()) {
     case 2: return rforward<nn::Sequential>(m, k->f.r(), x[0], x[1]);
     case 3: return rforward<nn::Sequential>(m, k->f.r(), x[0], x[1], x[2]);
     case 4: return rforward<nn::Sequential>(m, k->f.r(), x[0], x[1], x[2], x[3]);
     case 5: return k->f.in()==Cast::attention
                    ? rforward<nn::Sequential>(m, k->f.r(), x[0], x[1], x[2], x[3], x[4].item<bool>())
                    : rforward<nn::Sequential>(m, k->f.r(), x[0], x[1], x[2], x[3], x[4]);
     case 6: return k->f.in()==Cast::attention
                    ? rforward<nn::Sequential>(m, k->f.r(), x[0], x[1], x[2], x[3], x[4].item<bool>(), x[5])
                    : rforward<nn::Sequential>(m, k->f.r(), x[0], x[1], x[2], x[3], x[4], x[5]);
     case 7: return rforward<nn::Sequential>(m, k->f.r(), x[0], x[1], x[2], x[3], x[4], x[5], x[6]);
     case 8: return rforward<nn::Sequential>(m, k->f.r(), x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]);
     default: TORCH_ERROR(msym(c),": no forward calculation implemented for ",x.size()," tensor(s)");
    }
  case Cast::callback:
    switch(x.size()) {
     case 2: return rforward<knn::Callback>(m, k->f.r(), x[0], x[1]);
     case 3: return rforward<knn::Callback>(m, k->f.r(), x[0], x[1], x[2]);
     default: TORCH_ERROR(msym(c),": no forward calculation implemented for ",x.size()," tensor(s)");
    }
  case Cast::attention:  // needs special case due to 3 or more tensor args & boolean(???) arg mix
   switch(x.size()) {
    case 3: return m.as<nn::MultiheadAttention>()->forward(x[0], x[1], x[2]);
    case 4: return m.as<nn::MultiheadAttention>()->forward(x[0], x[1], x[2], x[3]);
    case 5: return m.as<nn::MultiheadAttention>()->forward(x[0], x[1], x[2], x[3], x[4].item<bool>());
    case 6: return m.as<nn::MultiheadAttention>()->forward(x[0], x[1], x[2], x[3], x[4].item<bool>(), x[5]);
    default: TORCH_ERROR(msym(c),": no forward calculation implemented for ",x.size()," tensor(s)");
   }
  case Cast::bilinear:        return m.as<nn::Bilinear>()->forward(x[0],x[1]);
  case Cast::cat:             return m.as<knn::Cat>()->forward(x[0],x[1]);
  case Cast::gru:             return m.as<nn::GRU>()->forward(x[0],x[1]);
  case Cast::mul:             return m.as<knn::Mul>()->forward(x[0],x[1]);
  case Cast::matmul:          return m.as<knn::Matmul>()->forward(x[0],x[1]);
  case Cast::pairwise:        return m.as<nn::PairwiseDistance>()->forward(x[0],x[1]);
  case Cast::rnn:             return m.as<nn::RNN>()->forward(x[0],x[1]);
  case Cast::seqjoin:         return m.as<knn::SeqJoin>()->forward(x[0],x[1]);
  case Cast::similar:         return m.as<nn::CosineSimilarity>()->forward(x[0],x[1]);

  case Cast::seqnest:         return vforward3<knn::SeqNest>(c,m,x);
  case Cast::recur:           return vforward3<knn::Recur>(c,m,x);
  case Cast::residual:        return vforward3<knn::Residual>(c,m,x);
  case Cast::selfattention:   return vforward3<knn::SelfAttention>(c,m,x);
  case Cast::encoder:         return vforward3<nn::TransformerEncoder>(c,m,x);
  case Cast::encoderlayer:    return vforward3<nn::TransformerEncoderLayer>(c,m,x);

  case Cast::decoder:         return vforward6<nn::TransformerDecoder>(c,m,x);
  case Cast::decoderlayer:    return vforward6<nn::TransformerDecoderLayer>(c,m,x);
  case Cast::transformer:     return vforward8<nn::Transformer>(c,m,x);
  case Cast::seqdict:         return vforward8<knn::SeqDict>(c,m,x);
  case Cast::seqlist:         return vforward8<knn::SeqList>(c,m,x);
  default: TORCH_ERROR(msym(c),": no forward calculation implemented for ",x.size()," tensor(s)");
 }
}

// ------------------------------------------------------------------------
// vforward - handle vector input for various type of forward calculations
// ------------------------------------------------------------------------
static Output vforward(Kmodule *m,const TensorVector& v) {
 auto f=m->f; auto c=m->c, i=f.in();
 auto a=f.a(); auto n=f.n(); auto an=a.size(); auto vn=v.size();
 if(an==1) {
   switch(a.front()) {
    case Arg::tensor:
     TORCH_CHECK(vn==1, msym(i),": forward requires a single tensor, ",vn," tensors given");
     tforward(c,m,v.front());
    case Arg::vector:
     return vforward1(c, f.r(), m->module(), v);
    case Arg::tuple:
     TORCH_CHECK(vn==2, msym(i),": forward calc requires tuple of 2 tensors, ",vn," given");
     return xforward(c, m, std::make_tuple(v[0],v[1]));
    case Arg::nested:
     TORCH_CHECK(vn==3, msym(i),": forward calc requires nested tuple of 3 tensors, ",vn," given");
     return xforward(c, m, std::make_tuple(v[0], std::make_tuple(v[1],v[2])));
    default:
     TORCH_ERROR(msym(i),": forward not implemented for ",argname(a.front()));
   }
 } else if(c==Cast::attention || tensorargs(a)) {  
   TORCH_CHECK(n<=vn,  msym(i),": ",vn," tensor(s) supplied, but at least ",n," required");
   TORCH_CHECK(vn<=an, msym(i),": ",vn," tensors supplied, but only ",an," expected");
   if(vn==1)
    return tforward(c,m,v.front());
   else
    return vforward(c,m,v);
 } else if(an==2 && a.front()==Arg::tensor && a.back()==Arg::tuple) {
   TORCH_CHECK(vn==3, msym(i),": expects 3 tensors for (tensor,tuple) input but ",vn," given");
   return xforward(c, m, v[0], std::make_tuple(v[1],v[2]));
 } else {
   TORCH_ERROR(msym(i),": forward calculation with ",argstring(a)," input is not implemented");
 }
}

// -----------------------------------------------------------------------------
// mforward - accepts k api module & input(s), determines form of forward() call
// -----------------------------------------------------------------------------
Output mforward(Kmodule *m,const Input& x) {
 const auto& f=m->f; const auto& a=f.a(); Cast c=m->c,i=f.in(); auto n=f.n(); auto an=a.size();
 TORCH_CHECK(f.f(), msym(c),": no forward calculation defined");
 TORCH_CHECK(an, msym(c),": unable to run forward calculation, no argument types defined");
 if(auto *p=c10::get_if<Tensor>(&x)) {
  TORCH_CHECK(n==1, msym(i),": ",n," args required for forward calculation, single tensor supplied");
  TORCH_CHECK(a.front()==Arg::tensor, msym(i),": argument of ",argname(a.front())," expected, but tensor supplied");
  return tforward(c,m,*p);
 } else if(auto *p=c10::get_if<TensorVector>(&x)) {
  return vforward(m,*p);
 } else {
  TORCH_ERROR(msym(i),": forward(",argstring(a),") not implemented given ",inputname(x));
 }
}

// -------------------------------------------------------------------
// normalize - pytorch has functional form only, no module as of v1.10
// fold & unfold: functional form of the Fold & Unfold modules
// interpolate - no module, pytorch functional form only
// linear,bilinear - invoke functional form
// -------------------------------------------------------------------
KAPI normalize(K x) {
 KTRY
  Tensor r,*t=nullptr;
  if(x->t || (t=xten(x))) {
   return kresult(t, fnn::normalize(t ? *t : kput(x), fnn::NormalizeFuncOptions()));
  } else {
   t=xten(x,0);
   return kresult(t||r.defined(), fnn::normalize(t ? *t : kput(x,0), knn::normalize(x,1,Cast::normalize,r)));
  }
 KCATCH("normalize");
}

static K kfold(K x,Cast c) {
 KTRY
  TORCH_CHECK(!x->t, msym(c)," not implemented for ",kname(x->t));
  Tensor r, *t=xten(x,0);
  return kresult(t, c==Cast::fold
       ? fnn::fold  (t ? *t : kput(x,0),   knn::fold(x,1,c))
       : fnn::unfold(t ? *t : kput(x,0), knn::unfold(x,1,c)));
 KCATCH("fold");
}

KAPI   fold(K x) {return kfold(x, Cast::fold);}
KAPI unfold(K x) {return kfold(x, Cast::unfold);}

KAPI interpolate(K x) {
 KTRY
  TORCH_CHECK(!x->t, "interpolate not implemented for ",kname(x->t));
  Tensor r, *t=xten(x,0);
  return kresult(t,
                 fnn::interpolate(t ? *t : kput(x,0),
                 knn::upsample<fnn::InterpolateFuncOptions>(x,1,Cast::interpolate)));
 KCATCH("interpolate");
}

KAPI linear(K x) {
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

KAPI bilinear(K x) {
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

// -----------------------------------
// functional form of pooling methods:
// -----------------------------------
static K pool(K x,Cast c) {
 KTRY
  TORCH_CHECK(!x->t, msym(c)," not implemented for ",kname(x->t));
  Tensor r, *t=xten(x,0);
  switch(c) {
   case Cast::maxpool1d:  r=fnn::max_pool1d(t ? *t : kput(x,0), knn::maxpool<1>(x,1,c)); break;
   case Cast::maxpool2d:  r=fnn::max_pool2d(t ? *t : kput(x,0), knn::maxpool<2>(x,1,c)); break;
   case Cast::maxpool3d:  r=fnn::max_pool3d(t ? *t : kput(x,0), knn::maxpool<3>(x,1,c)); break;
   case Cast::avgpool1d:  r=fnn::avg_pool1d(t ? *t : kput(x,0), knn::avgpool<1>(x,1,c)); break;
   case Cast::avgpool2d:  r=fnn::avg_pool2d(t ? *t : kput(x,0), knn::avgpool<2>(x,1,c)); break;
   case Cast::avgpool3d:  r=fnn::avg_pool3d(t ? *t : kput(x,0), knn::avgpool<3>(x,1,c)); break;
   case Cast::adaptmax1d: r=fnn::adaptive_max_pool1d(t ? *t : kput(x,0), knn::adapt<1,nn::AdaptiveMaxPool1dOptions>(x,1,c)); break;
   case Cast::adaptmax2d: r=fnn::adaptive_max_pool2d(t ? *t : kput(x,0), knn::adapt<2,nn::AdaptiveMaxPool2dOptions>(x,1,c)); break;
   case Cast::adaptmax3d: r=fnn::adaptive_max_pool3d(t ? *t : kput(x,0), knn::adapt<3,nn::AdaptiveMaxPool3dOptions>(x,1,c)); break;
   case Cast::adaptavg1d: r=fnn::adaptive_avg_pool1d(t ? *t : kput(x,0), knn::adapt<1,nn::AdaptiveAvgPool1dOptions>(x,1,c)); break;
   case Cast::adaptavg2d: r=fnn::adaptive_avg_pool2d(t ? *t : kput(x,0), knn::adapt<2,nn::AdaptiveAvgPool2dOptions>(x,1,c)); break;
   case Cast::adaptavg3d: r=fnn::adaptive_avg_pool3d(t ? *t : kput(x,0), knn::adapt<3,nn::AdaptiveAvgPool3dOptions>(x,1,c)); break;
   case Cast::fmaxpool2d: r=fnn::fractional_max_pool2d(t ? *t : kput(x,0), knn::fpool<2>(x,1,c)); break;
   case Cast::fmaxpool3d: r=fnn::fractional_max_pool3d(t ? *t : kput(x,0), knn::fpool<3>(x,1,c)); break;
   case Cast::lppool1d:   r=fnn::lp_pool1d(t ? *t : kput(x,0), knn::lppool<1>(x,1,c)); break;
   case Cast::lppool2d:   r=fnn::lp_pool2d(t ? *t : kput(x,0), knn::lppool<2>(x,1,c)); break;
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

KAPI kpad(K x) {
 KTRY
  TORCH_CHECK(!x->t, "pad not implemented for ",kname(x->t));
  Tensor r, *t=xten(x,0);
  return kresult(t, fnn::pad(t ? *t : kput(x,0), knn::pad(x,1,Cast::pad)));
 KCATCH("pad");
}

// -------------------------------------------------------------------
// noarg:  activation fns w'out args,
//         logsigmoid, mish, sigmoid, silu, softsign, tanh, tanhshrink
// -------------------------------------------------------------------
static void noarg(Cast c,K x,J i) {
 TORCH_CHECK(xnone(x,i), msym(c), ": no arguments expected, ", kstring(x));
}

static K noarg(const char* s,Tensor (*f)(const Tensor&), K x) {
 KTRY
  Tensor *t=xten(x); return kresult(t, f(t ? *t : kput(x)));
 KCATCH(s);
}

// v1.12.0 introduced optional string_view argument to torch::gelu
static Tensor kgelu(const Tensor& t) {return torch::gelu(t);}

KAPI gelu(K x)       {return noarg("gelu",       kgelu,              x);}
KAPI logsigmoid(K x) {return noarg("logsigmoid", torch::log_sigmoid, x);}
KAPI mish(K x)       {return noarg("mish",       fnn::mish,          x);}
KAPI silu(K x)       {return noarg("silu",       fnn::silu,          x);}
KAPI softsign(K x)   {return noarg("softsign",   fnn::softsign,      x);}
KAPI tanhshrink(K x) {return noarg("tanhshrink", fnn::tanhshrink,    x);}

KAPI onehot(K x) {  // functional invocation w'additional args for no. of classes,optional datatype
 KTRY
  knn::OneHotOptions o; Tensor *t=xten(x);
  if(t)
   return kten(knn::onehot(*t,o));
  else if((t=xten(x,0)))
   return kten(knn::onehot(*t,knn::onehot(x,1,Cast::onehot)));
  else if(xarray(x,3))
   return kget(knn::onehot(kput(x),o));
  else
   return kget(knn::onehot(kput(x,0),knn::onehot(x,1,Cast::onehot)));
 KCATCH("onehot");
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
  else if(xarray(x,3)) p=false,a=false,t=kput(x);
  else                 p=false,a=true, t=kput(x,0);
  switch(c) {
   case Cast::relu:  r=fnn::relu (t,a ? knn::inplace(x,1,c) : false); break;
   case Cast::relu6: r=fnn::relu6(t,a ? knn::inplace(x,1,c) : false); break;
   case Cast::selu:  r=fnn::selu (t,a ? knn::inplace(x,1,c) : false); break;
   case Cast::elu:   r=fnn::elu(t,knn::alpha<nn::ELUOptions>(a ? x : nullptr,1,c)); break;
   case Cast::celu:  r=fnn::celu(t,knn::alpha<nn::CELUOptions>(a ? x : nullptr,1,c)); break;
   case Cast::leakyrelu: r=fnn::leaky_relu(t,knn::slope(a ? x : nullptr,1,c)); break;
   case Cast::hardshrink: r=torch::hardshrink(t,a ? knn::lambda(x,1,c) : knn::lambda(c)); break;
   case Cast::softshrink: r=torch::softshrink(t,a ? knn::lambda(x,1,c) : knn::lambda(c)); break;
   case Cast::glu:        r=fnn::glu(t,a ? knn::dim(x,1,c) : knn::dim(c)); break;
   case Cast::softmin:
   case Cast::softmax:
   case Cast::logsoftmax: {
    auto d=knn::softdim(t.dim()); c10::optional<Dtype> s; if(a) knn::softargs(x,1,c,d,s);
    switch(c) {
     case Cast::softmin:    r=fnn::detail::softmin(t,d,s); break;
     case Cast::softmax:    r=fnn::detail::softmax(t,d,s); break;
     case Cast::logsoftmax: r=fnn::detail::log_softmax(t,d,s); break;
     default: TORCH_ERROR("unrecognized activation function");
    }
    break;
   }
   case Cast::rrelu: {
    double lo,up; bool in,tr; knn::rrelu(a ? x : nullptr,1,c,false,tr,in,lo,up);
    r=fnn::detail::rrelu(t,lo,up,tr,in);
    break;
   }
   case Cast::hardtanh:  r=fnn::hardtanh (t, knn::hardtanh(a ? x : nullptr,1,c)); break;
   case Cast::softplus:  r=fnn::softplus (t, knn::softplus(a ? x : nullptr,1,c)); break;
   case Cast::threshold: r=fnn::threshold(t, knn::threshold(a ? x : nullptr,1,c)); break;
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
KAPI      rrelu(K x) {return act(x, Cast::rrelu,      "rrelu");}
KAPI   hardtanh(K x) {return act(x, Cast::hardtanh,   "hardtanh");}
KAPI   softplus(K x) {return act(x, Cast::softplus,   "softplus");}
KAPI  threshold(K x) {return act(x, Cast::threshold,  "threshold");}

KAPI prelu(K x) {
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

// ------------------------------------------------------------------------
// functional form of the distance calculations & pdist
// ------------------------------------------------------------------------
static K distance(K x,Cast c) {
 KTRY
  TORCH_CHECK(!x->t, msym(c)," not implemented for ",kname(x->t));
  Tensor r, *a=xten(x,0), *b=xten(x,1);
  switch(c) {
   case Cast::pairwise: r=fnn::pairwise_distance(a ? *a : kput(x,0), b ? *b : kput(x,1), knn::pairwise(x,2,c)); break;
   case Cast::similar:  r=fnn::cosine_similarity(a ? *a : kput(x,0), b ? *b : kput(x,1), knn::similar(x,2,c)); break;
   default: TORCH_ERROR("unrecognized distance function");
  }
  return kresult(a||b,r);
 KCATCH("distance");
}

KAPI pairwise(K x) {return distance(x,Cast::pairwise);}
KAPI similar(K x)  {return distance(x,Cast::similar);}

KAPI pdist(K x) {
 KTRY
  TORCH_CHECK(!x->t, "pdist not implemented for ",kname(x->t));
  F p=2; bool b=x->n==2 && xnum(x,1,p); Tensor *t = b ? xten(x,0) : xten(x);
  return kresult(t, torch::pdist(t ? *t : (b ? kput(x,0) : kput(x)), p));
 KCATCH("pdist");
}

// -----------------------------------------------------------------------
// flatten - functional form given input/tensor, optional start & end dims
// -----------------------------------------------------------------------
KAPI flatten(K x) {  
// functional invocation w'different defaults than module(start dim=0 not 1)
 KTRY
  Tensor *t=xten(x);
  if(t) {
   return kten(torch::flatten(*t));
  } else if(xarray(x,3)) {
   return kget(torch::flatten(kput(x)));
  } else {
   t=xten(x,0);
   auto o=knn::flatten(x,1,Cast::flatten,true);
   return kresult(t, torch::flatten(t ? *t : kput(x,0), o.start_dim(), o.end_dim()));
  }
 KCATCH("flatten");
}

// --------------------------------------------------------------------------
// transforms similar to pytorch vision transforms
//            Normailze, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip
// zscore - subtract given mean and divide by given standard deviation
// randomcrop - crop image at random location w'output size & padding options
// randomflip - flip image vertically or horizintally with give probability
// --------------------------------------------------------------------------
KAPI zscore(K x) {
 KTRY
  if(x->t) {
   TORCH_CHECK(x->t > 0, "zscore: not implemented for ",kname(x));
   Tensor t=kput(x);
   TORCH_CHECK(t.dim()>0 && t.size(0)==3, "zscore: expecting 3-element list, given ",x->n," element(s)");
   return kget(knn::zscore(t[0],t[1],t[2]));
  } else {
   Tensor *t=xten(x,0); const auto o=knn::zscore(x,1,Cast::zscore);
   if(t && o.inplace()) {
    return knn::zscore_(*t, o.mean(), o.stddev()), (K)0;
   } else {
    return kresult(t, knn::zscore(t ? *t : kput(x,0), o.mean(), o.stddev()));
   }
  }
 KCATCH("zscore");
}

KAPI randomcrop(K x) {
 KTRY
  TORCH_CHECK(!x->t, "randomcrop: not implemented for ",kname(x));
  TORCH_CHECK(x->n>1 && x->n<6, "randomcrop: 2-5 args expected, (input;size;pad;padmode;value), but ",x->n," given");
  Tensor *t=xten(x,0);
  return kresult(t, knn::rcrop(t ? *t : kput(x,0), knn::rcrop(x,1,Cast::randomcrop)));
 KCATCH("randomcrop");
}

KAPI randomflip(K x) {
 KTRY
  TORCH_CHECK(!x->t, "randomflip: not implemented for ",kname(x));
  Tensor *t=xten(x); if(!t) t=xten(x,0);
  if(t || !xarray(x,3)) {
   return kresult(t, knn::rflip(t ? *t : kput(x,0), knn::rflip(x,1,Cast::randomflip)));
  } else {
   return kget(knn::rflip(kput(x), knn::RandomFlipOptions()));
  }
 KCATCH("randomflip");
}

// ----------------------------------------------------------------------------
// mcreate - define module from supplied options, return as generic module ptr
// ----------------------------------------------------------------------------
Moduleptr mcreate(K x,J i,Cast c) {
 switch(c) {
  case Cast::sequential:  noarg(c,x,i); return nn::Sequential().ptr();  //containers
  case Cast::seqdict:     noarg(c,x,i); return knn::SeqDict().ptr();
  case Cast::seqlist:     noarg(c,x,i); return knn::SeqList().ptr();
  case Cast::seqnest:     noarg(c,x,i); return knn::SeqNest().ptr();
  case Cast::seqjoin:     noarg(c,x,i); return knn::SeqJoin().ptr();
  case Cast::moduledict:  noarg(c,x,i); return nn::ModuleDict().ptr();
  case Cast::modulelist:  noarg(c,x,i); return nn::ModuleList().ptr();
  case Cast::fork:        noarg(c,x,i); return knn::Fork().ptr();
  case Cast::residual:    noarg(c,x,i); return knn::Residual().ptr();
  case Cast::transform:   noarg(c,x,i); return knn::Transform().ptr();
  case Cast::nbeats:      noarg(c,x,i); return knn::NBeats().ptr();
  case Cast::parmdict:    return parmdict(x,i); // dictionary can contain parms as options

  case Cast::callback:    return knn::callback(x,i,c);

  case Cast::batchnorm1d:  return nn::BatchNorm1d(knn::batchnorm<nn::BatchNormOptions>(x,i,c)).ptr();
  case Cast::batchnorm2d:  return nn::BatchNorm2d(knn::batchnorm<nn::BatchNormOptions>(x,i,c)).ptr();
  case Cast::batchnorm3d:  return nn::BatchNorm3d(knn::batchnorm<nn::BatchNormOptions>(x,i,c)).ptr();

  case Cast::instancenorm1d:  return nn::InstanceNorm1d(knn::batchnorm<nn::InstanceNormOptions>(x,i,c)).ptr();
  case Cast::instancenorm2d:  return nn::InstanceNorm2d(knn::batchnorm<nn::InstanceNormOptions>(x,i,c)).ptr();
  case Cast::instancenorm3d:  return nn::InstanceNorm3d(knn::batchnorm<nn::InstanceNormOptions>(x,i,c)).ptr();

  case Cast::groupnorm:  return nn::GroupNorm(knn::groupnorm(x,i,c)).ptr();
  case Cast::layernorm:  return nn::LayerNorm(knn::layernorm(x,i,c)).ptr();
  case Cast::localnorm:  return nn::LocalResponseNorm(knn::localnorm<nn::LocalResponseNormOptions>(x,i,c)).ptr();
  case Cast::crossmap2d: return nn::CrossMapLRN2d(knn::localnorm<nn::CrossMapLRN2dOptions>(x,i,c)).ptr();

  case Cast::embed:        return knn::embed(x,i,c).ptr();
  case Cast::embedbag:     return knn::embedbag(x,i,c).ptr();
  case Cast::embedpos:     return knn::EmbedPosition(knn::embedpos(x,i,c)).ptr();
  case Cast::embedseq:     return knn::EmbedSequence(knn::embedseq(x,i,c)).ptr();
  case Cast::linear:       return nn::Linear(knn::linear(x,i,c)).ptr();
  case Cast::bilinear:     return nn::Bilinear(knn::bilinear(x,i,c)).ptr();

  case Cast::drop:         return nn::Dropout(knn::drop(x,i,c)).ptr();
  case Cast::drop2d:       return nn::Dropout2d(knn::drop(x,i,c)).ptr();
  case Cast::drop3d:       return nn::Dropout3d(knn::drop(x,i,c)).ptr();
  case Cast::droppath:     return knn::DropPath(knn::droppath(x,i,c)).ptr();
  case Cast::adrop:        return nn::AlphaDropout(knn::drop(x,i,c)).ptr();
  case Cast::fadrop:       return nn::FeatureAlphaDropout(knn::drop(x,i,c)).ptr();

  case Cast::conv1d:       return nn::Conv1d(knn::conv<1>(x,i,c)).ptr();
  case Cast::conv2d:       return nn::Conv2d(knn::conv<2>(x,i,c)).ptr();
  case Cast::conv3d:       return nn::Conv3d(knn::conv<3>(x,i,c)).ptr();

  case Cast::convtranspose1d:  return nn::ConvTranspose1d(knn::convtran<1>(x,i,c)).ptr();
  case Cast::convtranspose2d:  return nn::ConvTranspose2d(knn::convtran<2>(x,i,c)).ptr();
  case Cast::convtranspose3d:  return nn::ConvTranspose3d(knn::convtran<3>(x,i,c)).ptr();

  case Cast::fold:         return nn::Fold(knn::fold(x,i,c)).ptr();
  case Cast::unfold:       return nn::Unfold(knn::unfold(x,i,c)).ptr();
  case Cast::upsample:     return nn::Upsample(knn::upsample<nn::UpsampleOptions>(x,i,c)).ptr();

  case Cast::maxpool1d:    return nn::MaxPool1d(knn::maxpool<1>(x,i,c)).ptr();
  case Cast::maxpool2d:    return nn::MaxPool2d(knn::maxpool<2>(x,i,c)).ptr();
  case Cast::maxpool3d:    return nn::MaxPool3d(knn::maxpool<3>(x,i,c)).ptr();

  case Cast::avgpool1d:    return nn::AvgPool1d(knn::avgpool<1>(x,i,c)).ptr();
  case Cast::avgpool2d:    return nn::AvgPool2d(knn::avgpool<2>(x,i,c)).ptr();
  case Cast::avgpool3d:    return nn::AvgPool3d(knn::avgpool<3>(x,i,c)).ptr();

  case Cast::adaptmax1d:   return nn::AdaptiveMaxPool1d(knn::adapt<1,nn::AdaptiveMaxPool1dOptions>(x,i,c)).ptr();
  case Cast::adaptmax2d:   return nn::AdaptiveMaxPool2d(knn::adapt<2,nn::AdaptiveMaxPool2dOptions>(x,i,c)).ptr();
  case Cast::adaptmax3d:   return nn::AdaptiveMaxPool3d(knn::adapt<3,nn::AdaptiveMaxPool3dOptions>(x,i,c)).ptr();

  case Cast::adaptavg1d:   return nn::AdaptiveAvgPool1d(knn::adapt<1,nn::AdaptiveAvgPool1dOptions>(x,i,c)).ptr();
  case Cast::adaptavg2d:   return nn::AdaptiveAvgPool2d(knn::adapt<2,nn::AdaptiveAvgPool2dOptions>(x,i,c)).ptr();
  case Cast::adaptavg3d:   return nn::AdaptiveAvgPool3d(knn::adapt<3,nn::AdaptiveAvgPool3dOptions>(x,i,c)).ptr();

  case Cast::fmaxpool2d:   return nn::FractionalMaxPool2d(knn::fpool<2>(x,i,c)).ptr();
  case Cast::fmaxpool3d:   return nn::FractionalMaxPool3d(knn::fpool<3>(x,i,c)).ptr();

  case Cast::lppool1d:     return nn::LPPool1d(knn::lppool<1>(x,i,c)).ptr();
  case Cast::lppool2d:     return nn::LPPool2d(knn::lppool<2>(x,i,c)).ptr();

  case Cast::pad:          return knn::Pad(knn::pad(x,i,c)).ptr();
  case Cast::pad1d:        return nn::ConstantPad1d(knn::cpad<1,nn::ConstantPad1dOptions>(x,i,c)).ptr();
  case Cast::pad2d:        return nn::ConstantPad2d(knn::cpad<2,nn::ConstantPad2dOptions>(x,i,c)).ptr();
  case Cast::pad3d:        return nn::ConstantPad3d(knn::cpad<3,nn::ConstantPad3dOptions>(x,i,c)).ptr();
  case Cast::reflect1d:    return nn::ReflectionPad1d(knn::npad<1,nn::ReflectionPad1dOptions>(x,i,c)).ptr();
  case Cast::reflect2d:    return nn::ReflectionPad2d(knn::npad<2,nn::ReflectionPad2dOptions>(x,i,c)).ptr();
  case Cast::replicate1d:  return nn::ReplicationPad1d(knn::npad<1,nn::ReplicationPad1dOptions>(x,i,c)).ptr();
  case Cast::replicate2d:  return nn::ReplicationPad2d(knn::npad<2,nn::ReplicationPad2dOptions>(x,i,c)).ptr();
  case Cast::replicate3d:  return nn::ReplicationPad3d(knn::npad<3,nn::ReplicationPad3dOptions>(x,i,c)).ptr();
  case Cast::zeropad2d:    return nn::ZeroPad2d(knn::npad<2,nn::ZeroPad2dOptions>(x,i,c)).ptr();

  case Cast::attention:     return nn::MultiheadAttention(knn::attention(x,i,c)).ptr();
  case Cast::selfattention: return knn::SelfAttention(knn::selfattn(x,i,c)).ptr();
  case Cast::decoderlayer:  return nn::TransformerDecoderLayer(knn::codelayer<nn::TransformerDecoderLayerOptions>(x,i,c)).ptr();
  case Cast::encoderlayer:  return nn::TransformerEncoderLayer(knn::codelayer<nn::TransformerEncoderLayerOptions>(x,i,c)).ptr();
  case Cast::decoder:       return nn::TransformerDecoder(knn::decoder(x,i,c)).ptr();
  case Cast::encoder:       return nn::TransformerEncoder(knn::encoder(x,i,c)).ptr();
  case Cast::transformer:   return nn::Transformer(knn::transformer(x,i,c)).ptr();

  case Cast::rnn:          return nn::RNN(knn::rnn(x,i,c)).ptr();
  case Cast::gru:          return nn::GRU(knn::rnn<nn::GRUOptions>(x,i,c)).ptr();
  case Cast::lstm:         return nn::LSTM(knn::rnn<nn::LSTMOptions>(x,i,c)).ptr();
  case Cast::recur:        return knn::Recur(knn::recur(x,i,c)).ptr();

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
  case Cast::mul:          noarg(c,x,i); return knn::Mul().ptr();
  case Cast::matmul:       noarg(c,x,i); return knn::Matmul().ptr();

  case Cast::relu:         return  nn::ReLU(knn::inplace(x,i,c)).ptr();
  case Cast::relu6:        return nn::ReLU6(knn::inplace(x,i,c)).ptr();
  case Cast::selu:         return  nn::SELU(knn::inplace(x,i,c)).ptr();

  case Cast::softmax:      return nn::Softmax(knn::dim(x,i,c)).ptr();
  case Cast::softmin:      return nn::Softmin(knn::dim(x,i,c)).ptr();
  case Cast::logsoftmax:   return nn::LogSoftmax(knn::dim(x,i,c)).ptr();
  case Cast::flatten:      return nn::Flatten(knn::flatten(x,i,c)).ptr();

  case Cast::select:       return knn::Select(knn::select(x,i,c)).ptr();
  case Cast::indexselect:  return knn::IndexSelect(knn::indexselect(x,i,c)).ptr();
  case Cast::squeeze:      return knn::Squeeze(knn::squeeze(x,i,c)).ptr();
  case Cast::unsqueeze:    return knn::Unsqueeze(knn::squeeze(x,i,c)).ptr();
  case Cast::expand:       return knn::Expand(knn::getsize(x,i,c)).ptr();
  case Cast::permute:      return knn::Permute(knn::getsize(x,i,c)).ptr();
  case Cast::reshape:      return knn::Reshape(knn::getsize(x,i,c)).ptr();
  case Cast::cat:          return knn::Cat(knn::dim(x,i,c)).ptr();
  case Cast::onehot:       return knn::OneHot(knn::onehot(x,i,c)).ptr();
  case Cast::transpose:    return knn::Transpose(knn::transpose(x,i,c)).ptr();

  case Cast::elu:          return nn::ELU (knn::alpha<nn::ELUOptions> (x,i,c)).ptr();
  case Cast::celu:         return nn::CELU(knn::alpha<nn::CELUOptions>(x,i,c)).ptr();
  case Cast::leakyrelu:    return nn::LeakyReLU(knn::slope(x,i,c)).ptr();
  case Cast::glu:          return nn::GLU(knn::dim(x,i,c)).ptr();
  case Cast::hardshrink:   return nn::Hardshrink(knn::lambda(x,i,c)).ptr();
  case Cast::softshrink:   return nn::Softshrink(knn::lambda(x,i,c)).ptr();
  case Cast::prelu:        return nn::PReLU(knn::prelu(x,i,c)).ptr();
  case Cast::rrelu:        return nn::RReLU(knn::rrelu(x,i,c)).ptr();
  case Cast::hardtanh:     return nn::Hardtanh(knn::hardtanh(x,i,c)).ptr();
  case Cast::softplus:     return nn::Softplus(knn::softplus(x,i,c)).ptr();
  case Cast::threshold:    return nn::Threshold(knn::threshold(x,i,c)).ptr();
  case Cast::pairwise:     return nn::PairwiseDistance(knn::pairwise(x,i,c)).ptr();
  case Cast::similar:      return nn::CosineSimilarity(knn::similar(x,i,c)).ptr();

  case Cast::zscore:       return knn::Zscore(knn::zscore(x,i,c)).ptr();
  case Cast::randomcrop:   return knn::RandomCrop(knn::rcrop(x,i,c)).ptr();
  case Cast::randomflip:   return knn::RandomFlip(knn::rflip(x,i,c)).ptr();
  default:
   if(container(c))
    TORCH_ERROR("cannot create container module: ",msym(c));
   else
    TORCH_ERROR("unrecognized module: cannot create module from unrecognized enumeration ",(I)c);
 }
}

// ---------------------------------------------------------------------------------------
// anymodule - given generic ptr, recast to specific type and return type-erased AnyModule
// ---------------------------------------------------------------------------------------
#define ANYMODULE(x,y) AnyModule(std::dynamic_pointer_cast<x>(y))
#define ANY(x,y) ANYMODULE(x##Impl,y)

AnyModule anymodule(Cast c,const Moduleptr& m) {
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
  case Cast::callback:        return m->as<knn::Callback>()->any();
  case Cast::batchnorm1d:     return ANY(nn::BatchNorm1d, m);
  case Cast::batchnorm2d:     return ANY(nn::BatchNorm2d, m);
  case Cast::batchnorm3d:     return ANY(nn::BatchNorm3d, m);
  case Cast::bilinear:        return ANY(nn::Bilinear, m);
  case Cast::cat:             return ANY(knn::Cat, m);
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
  case Cast::droppath:        return ANY(knn::DropPath, m);
  case Cast::elu:             return ANY(nn::ELU, m);
  case Cast::embed:           return ANY(nn::Embedding, m);
  case Cast::embedbag:        return ANY(nn::EmbeddingBag, m);
  case Cast::embedpos:        return ANY(knn::EmbedPosition, m);
  case Cast::embedseq:        return ANY(knn::EmbedSequence, m);
  case Cast::encoder:         return ANY(nn::TransformerEncoder, m);
  case Cast::encoderlayer:    return ANY(nn::TransformerEncoderLayer, m);
  case Cast::expand:          return ANY(knn::Expand, m);
  case Cast::fadrop:          return ANY(nn::FeatureAlphaDropout, m);
  case Cast::flatten:         return ANY(nn::Flatten, m);
  case Cast::fmaxpool2d:      return ANY(nn::FractionalMaxPool2d, m);
  case Cast::fmaxpool3d:      return ANY(nn::FractionalMaxPool3d, m);
  case Cast::fold:            return ANY(nn::Fold, m);
  case Cast::fork:            return ANY(knn::Fork, m);
  case Cast::gelu:            return ANY(nn::GELU, m);
  case Cast::glu:             return ANY(nn::GLU, m);
  case Cast::groupnorm:       return ANY(nn::GroupNorm, m);
  case Cast::gru:             return ANY(nn::GRU, m);
  case Cast::hardshrink:      return ANY(nn::Hardshrink, m);
  case Cast::hardtanh:        return ANY(nn::Hardtanh, m);
  case Cast::identity:        return ANY(nn::Identity, m);
  case Cast::indexselect:     return ANY(knn::IndexSelect, m);
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
  case Cast::lstm:            return ANY(nn::LSTM, m);
  case Cast::matmul:          return ANY(knn::Matmul, m);
  case Cast::maxpool1d:       return ANY(nn::MaxPool1d, m);
  case Cast::maxpool2d:       return ANY(nn::MaxPool2d, m);
  case Cast::maxpool3d:       return ANY(nn::MaxPool3d, m);
  case Cast::mish:            return ANY(nn::Mish, m);
  case Cast::mul:             return ANY(knn::Mul, m);
  case Cast::nbeats:          return ANY(knn::NBeats, m);
  case Cast::onehot:          return ANY(knn::OneHot, m);
  case Cast::pad:             return ANY(knn::Pad, m);
  case Cast::pad1d:           return ANY(nn::ConstantPad1d, m);
  case Cast::pad2d:           return ANY(nn::ConstantPad2d, m);
  case Cast::pad3d:           return ANY(nn::ConstantPad3d, m);
  case Cast::pairwise:        return ANY(nn::PairwiseDistance, m);
  case Cast::permute:         return ANY(knn::Permute, m);
  case Cast::prelu:           return ANY(nn::PReLU, m);
  case Cast::randomcrop:      return ANY(knn::RandomCrop, m);
  case Cast::randomflip:      return ANY(knn::RandomFlip, m);
  case Cast::recur:           return ANY(knn::Recur, m);
  case Cast::reflect1d:       return ANY(nn::ReflectionPad1d, m);
  case Cast::reflect2d:       return ANY(nn::ReflectionPad2d, m);
  case Cast::relu:            return ANY(nn::ReLU, m);
  case Cast::relu6:           return ANY(nn::ReLU6, m);
  case Cast::replicate1d:     return ANY(nn::ReplicationPad1d, m);
  case Cast::replicate2d:     return ANY(nn::ReplicationPad2d, m);
  case Cast::replicate3d:     return ANY(nn::ReplicationPad3d, m);
  case Cast::residual:        return ANY(knn::Residual, m);
  case Cast::reshape:         return ANY(knn::Reshape, m);
  case Cast::rnn:             return ANY(nn::RNN, m);
  case Cast::rrelu:           return ANY(nn::RReLU, m);
  case Cast::select:          return ANY(knn::Select, m);
  case Cast::selfattention:   return ANY(knn::SelfAttention, m);
  case Cast::selu:            return ANY(nn::SELU, m);
  case Cast::seqjoin:         return ANY(knn::SeqJoin, m);
  case Cast::seqdict:         return ANY(knn::SeqDict, m);
  case Cast::seqlist:         return ANY(knn::SeqList, m);
  case Cast::seqnest:         return ANY(knn::SeqNest, m);
  case Cast::sigmoid:         return ANY(nn::Sigmoid, m);
  case Cast::silu:            return ANY(nn::SiLU, m);
  case Cast::similar:         return ANY(nn::CosineSimilarity, m);
  case Cast::softmax:         return ANY(nn::Softmax, m);
  case Cast::softmax2d:       return ANY(nn::Softmax2d, m);
  case Cast::softmin:         return ANY(nn::Softmin, m);
  case Cast::softplus:        return ANY(nn::Softplus, m);
  case Cast::softshrink:      return ANY(nn::Softshrink, m);
  case Cast::softsign:        return ANY(nn::Softsign, m);
  case Cast::squeeze:         return ANY(knn::Squeeze, m);
  case Cast::tanh:            return ANY(nn::Tanh, m);
  case Cast::tanhshrink:      return ANY(nn::Tanhshrink, m);
  case Cast::threshold:       return ANY(nn::Threshold, m);
  case Cast::transform:       return ANY(knn::Transform, m);
  case Cast::transformer:     return ANY(nn::Transformer, m);
  case Cast::transpose:       return ANY(knn::Transpose, m);
  case Cast::unfold:          return ANY(nn::Unfold, m);
  case Cast::unsqueeze:       return ANY(knn::Unsqueeze, m);
  case Cast::upsample:        return ANY(nn::Upsample, m);
  case Cast::zeropad2d:       return ANY(nn::ZeroPad2d, m);
  case Cast::zscore:          return ANY(knn::Zscore, m);

  case Cast::interpolate:
  case Cast::normalize:       TORCH_ERROR(msym(c),": unable to create type-erased module, only functional form implemented");
  case Cast::moduledict:
  case Cast::modulelist:
  case Cast::parmdict:        TORCH_ERROR(msym(c),": unable to create type-erased module, no forward method defined");
  case Cast::sequential:      TORCH_ERROR(msym(c),": unable to create type-erased module, forward method uses template");

  default: TORCH_ERROR("can't create type-erased module, unrecognized cast: ",(I)c);
 }
}

// ---------------------------------------------------------------------------
// moduleoptions - given enum & generic module, return options as k dictionary
// ---------------------------------------------------------------------------
K moduleoptions(bool a,bool b,Cast c,const Module& m) {
// a:true for all options, else only non-default, b:true if retrieving state
 switch(c) {
  case Cast::sequential:      //container modules w'out options
  case Cast::seqdict:
  case Cast::seqlist:
  case Cast::seqnest:
  case Cast::seqjoin:
  case Cast::moduledict:
  case Cast::modulelist:
  case Cast::fork:
  case Cast::residual:
  case Cast::transform:
  case Cast::nbeats:
  case Cast::gelu:            //pointwise activation fns w'out options
  case Cast::identity:
  case Cast::logsigmoid:
  case Cast::mish:
  case Cast::mul:
  case Cast::matmul:
  case Cast::sigmoid:
  case Cast::silu:
  case Cast::softsign:
  case Cast::softmax2d:
  case Cast::tanh:
  case Cast::tanhshrink:
   return KDICT;

  case Cast::parmdict: return b ? KDICT : kget(m.named_parameters());  // return parms as options if no state required
  case Cast::callback: return callback(a,b,*m.as<knn::Callback>());
  
  case Cast::batchnorm1d:      return knn::batchnorm(a,m.as<nn::BatchNorm1d>()->options);
  case Cast::batchnorm2d:      return knn::batchnorm(a,m.as<nn::BatchNorm2d>()->options);
  case Cast::batchnorm3d:      return knn::batchnorm(a,m.as<nn::BatchNorm3d>()->options);
  case Cast::instancenorm1d:   return knn::batchnorm(a,m.as<nn::InstanceNorm1d>()->options);
  case Cast::instancenorm2d:   return knn::batchnorm(a,m.as<nn::InstanceNorm2d>()->options);
  case Cast::instancenorm3d:   return knn::batchnorm(a,m.as<nn::InstanceNorm3d>()->options);
  case Cast::groupnorm:        return knn::groupnorm(a,m.as<nn::GroupNorm>()->options);
  case Cast::layernorm:        return knn::layernorm(a,m.as<nn::LayerNorm>()->options);
  case Cast::localnorm:        return knn::localnorm(a,c,m.as<nn::LocalResponseNorm>()->options);
  case Cast::crossmap2d:       return knn::localnorm(a,c,m.as<nn::CrossMapLRN2d>()->options);

  case Cast::embed:    {auto* e=m.as<nn::Embedding>();    return knn::embed(a,c,e->options,e->weight);}
  case Cast::embedbag: {auto* e=m.as<nn::EmbeddingBag>(); return knn::embed(a,c,e->options,e->weight);}
  case Cast::embedpos: return knn::embedpos(a,m.as<knn::EmbedPosition>()->options);
  case Cast::embedseq: return knn::embedseq(a,m.as<knn::EmbedSequence>()->options);

  case Cast::linear:           return knn::linear(a,m.as<nn::Linear>()->options);
  case Cast::bilinear:         return knn::bilinear(a,m.as<nn::Bilinear>()->options);

  case Cast::drop:             return knn::drop(a,m.as<nn::Dropout>()->options);
  case Cast::drop2d:           return knn::drop(a,m.as<nn::Dropout2d>()->options);
  case Cast::drop3d:           return knn::drop(a,m.as<nn::Dropout3d>()->options);
  case Cast::droppath:         return knn::droppath(a,m.as<knn::DropPath>()->options);
  case Cast::adrop:            return knn::drop(a,m.as<nn::AlphaDropout>()->options);
  case Cast::fadrop:           return knn::drop(a,m.as<nn::FeatureAlphaDropout>()->options);

  case Cast::conv1d:           return knn::conv(a,m.as<nn::Conv1d>()->options);
  case Cast::conv2d:           return knn::conv(a,m.as<nn::Conv2d>()->options);
  case Cast::conv3d:           return knn::conv(a,m.as<nn::Conv3d>()->options);
  case Cast::convtranspose1d:  return knn::conv(a,m.as<nn::ConvTranspose1d>()->options);
  case Cast::convtranspose2d:  return knn::conv(a,m.as<nn::ConvTranspose2d>()->options);
  case Cast::convtranspose3d:  return knn::conv(a,m.as<nn::ConvTranspose3d>()->options);

  case Cast::fold:             return knn::fold(a,m.as<nn::Fold>()->options);
  case Cast::unfold:           return knn::unfold(a,m.as<nn::Unfold>()->options);
  case Cast::upsample:         return knn::upsample(a,m.as<nn::Upsample>()->options);

  case Cast::maxpool1d:        return knn::maxpool(a,m.as<nn::MaxPool1d>()->options);
  case Cast::maxpool2d:        return knn::maxpool(a,m.as<nn::MaxPool2d>()->options);
  case Cast::maxpool3d:        return knn::maxpool(a,m.as<nn::MaxPool3d>()->options);

  case Cast::avgpool1d:        return knn::avgpool(a,m.as<nn::AvgPool1d>()->options);
  case Cast::avgpool2d:        return knn::avgpool(a,m.as<nn::AvgPool2d>()->options);
  case Cast::avgpool3d:        return knn::avgpool(a,m.as<nn::AvgPool3d>()->options);

  case Cast::adaptmax1d:       return knn::adapt(m.as<nn::AdaptiveMaxPool1d>()->options);
  case Cast::adaptmax2d:       return knn::adapt(m.as<nn::AdaptiveMaxPool2d>()->options);
  case Cast::adaptmax3d:       return knn::adapt(m.as<nn::AdaptiveMaxPool3d>()->options);

  case Cast::adaptavg1d:       return knn::adapt(m.as<nn::AdaptiveAvgPool1d>()->options);
  case Cast::adaptavg2d:       return knn::adapt(m.as<nn::AdaptiveAvgPool2d>()->options);
  case Cast::adaptavg3d:       return knn::adapt(m.as<nn::AdaptiveAvgPool3d>()->options);

  case Cast::fmaxpool2d:       return knn::fpool(a,m.as<nn::FractionalMaxPool2d>()->options);
  case Cast::fmaxpool3d:       return knn::fpool(a,m.as<nn::FractionalMaxPool3d>()->options);

  case Cast::lppool1d:         return knn::lppool(a,m.as<nn::LPPool1d>()->options);
  case Cast::lppool2d:         return knn::lppool(a,m.as<nn::LPPool2d>()->options);

  case Cast::pad:              return knn::pad(a,m.as<knn::Pad>()->options);
  case Cast::pad1d:            return knn::cpad(m.as<nn::ConstantPad1d>()->options);
  case Cast::pad2d:            return knn::cpad(m.as<nn::ConstantPad2d>()->options);
  case Cast::pad3d:            return knn::cpad(m.as<nn::ConstantPad3d>()->options);
  case Cast::reflect1d:        return knn::npad(m.as<nn::ReflectionPad1d>()->options);
  case Cast::reflect2d:        return knn::npad(m.as<nn::ReflectionPad2d>()->options);
  case Cast::replicate1d:      return knn::npad(m.as<nn::ReplicationPad1d>()->options);
  case Cast::replicate2d:      return knn::npad(m.as<nn::ReplicationPad2d>()->options);
  case Cast::replicate3d:      return knn::npad(m.as<nn::ReplicationPad3d>()->options);
  case Cast::zeropad2d:        return knn::npad(m.as<nn::ZeroPad2d>()->options);

  case Cast::attention:        return knn::attention(a,m.as<nn::MultiheadAttention>()->options);
  case Cast::selfattention:    return knn::selfattn(a,m.as<knn::SelfAttention>()->options);
  case Cast::encoderlayer:     return resolvedict(knn::codelayer(a,c,m.as<nn::TransformerEncoderLayer>()->options));
  case Cast::decoderlayer:     return resolvedict(knn::codelayer(a,c,m.as<nn::TransformerDecoderLayer>()->options));
  case Cast::encoder:          return knn::encoder(a,c,m.as<nn::TransformerEncoder>()->options);
  case Cast::decoder:          return knn::decoder(a,c,m.as<nn::TransformerDecoder>()->options);
  case Cast::transformer:      return knn::transformer(a,c,m.as<nn::Transformer>()->options);

  case Cast::rnn:              return knn::rnn(a,m.as<nn::RNN>()->options);
  case Cast::gru:              return knn::rnn(a,m.as<nn::GRU>()->options);
  case Cast::lstm:             return knn::rnn(a,m.as<nn::LSTM>()->options);
  case Cast::recur:            return knn::recur(a,m.as<knn::Recur>()->options);

  case Cast::relu:             return knn::inplace(a,m.as<nn::ReLU>()->options.inplace());
  case Cast::selu:             return knn::inplace(a,m.as<nn::SELU>()->options.inplace());
  case Cast::relu6:            return knn::inplace(a,m.as<nn::ReLU6>()->options.inplace());

  case Cast::softmax:          return knn::dim(a,c,m.as<nn::Softmax>()->options.dim());
  case Cast::softmin:          return knn::dim(a,c,m.as<nn::Softmin>()->options.dim());
  case Cast::logsoftmax:       return knn::dim(a,c,m.as<nn::LogSoftmax>()->options.dim());
  case Cast::flatten:          return knn::flatten(a,m.as<nn::Flatten>()->options);

  case Cast::select:           return select(a,m.as<knn::Select>()->options);
  case Cast::indexselect:      return knn::indexselect(a,m.as<knn::IndexSelect>()->options);
  case Cast::squeeze:          return knn::squeeze(a,m.as<knn::Squeeze>()->options);
  case Cast::unsqueeze:        return knn::squeeze(a,m.as<knn::Unsqueeze>()->options);
  case Cast::expand:           return knn::getsize(a,m.as<knn::Expand>()->options);
  case Cast::permute:          return knn::getsize(a,m.as<knn::Permute>()->options);
  case Cast::reshape:          return knn::getsize(a,m.as<knn::Reshape>()->options);
  case Cast::cat:              return knn::dim(a,c,m.as<knn::Cat>()->options.dim());
  case Cast::onehot:           return knn::onehot(a,m.as<knn::OneHot>()->options);
  case Cast::transpose:        return knn::transpose(a,m.as<knn::Transpose>()->options);

  case Cast::elu:              return knn::alpha(a,m.as<nn::ELU>()->options);
  case Cast::celu:             return knn::alpha(a,m.as<nn::CELU>()->options);
  case Cast::leakyrelu:        return knn::slope(a,c,m.as<nn::LeakyReLU>()->options);
  case Cast::glu:              return knn::dim(a,c,m.as<nn::GLU>()->options.dim());
  case Cast::hardshrink:       return knn::lambda(a,c,m.as<nn::Hardshrink>()->options.lambda());
  case Cast::softshrink:       return knn::lambda(a,c,m.as<nn::Softshrink>()->options.lambda());

  case Cast::prelu:            return knn::prelu(a,m.as<nn::PReLU>()->options);
  case Cast::rrelu:            return knn::rrelu(a,m.as<nn::RReLU>()->options);
  case Cast::hardtanh:         return knn::hardtanh(a,m.as<nn::Hardtanh>()->options);
  case Cast::softplus:         return knn::softplus(a,m.as<nn::Softplus>()->options);
  case Cast::threshold:        return knn::threshold(a,m.as<nn::Threshold>()->options);
  case Cast::pairwise:         return knn::pairwise(a,m.as<nn::PairwiseDistance>()->options);
  case Cast::similar:          return knn::similar(a,m.as<nn::CosineSimilarity>()->options);

  case Cast::zscore:           return knn::zscore(a,m.as<knn::Zscore>()->options);
  case Cast::randomcrop:       return knn::rcrop(a,m.as<knn::RandomCrop>()->options);
  case Cast::randomflip:       return knn::rflip(a,m.as<knn::RandomFlip>()->options);

  default: TORCH_ERROR("unrecognized module: ",mlabel(m),", unable to retrieve options");
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
 const auto& a=anymodule(mcast(y),y);
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
 } else if(auto *m=x->as<knn::SeqNest>())   { addany(m,s,y);
 } else if(auto *m=x->as<knn::SeqJoin>())   { addseq(m,s,y);
 } else if(auto *m=x->as<knn::Fork>())      { addseq(m,s,y);
 } else if(auto *m=x->as<knn::Residual>())  { addseq(m,s,y);
 } else if(auto *m=x->as<knn::Transform>()) { addseq(m,s,y);
 } else if(auto *m=x->as<knn::NBeats>())    { m->push_back(y);
 } else if(auto *m=x->as<knn::Recur>())     { m->push_back(y);
 } else if(auto *m=x->as<nn::ModuleList>()) { m->push_back(y);
 } else if(auto *m=x->as<nn::ModuleDict>()) {
  m->update({{s ? s : c10::to_string(m->children().size()), y}});
 } else if(auto *m=x->as<knn::Callback>()) {
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
  K x=moduleoptions(true,false,v,m1),y=moduleoptions(true,false,w,m2),z;
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
 Modules q=mstack(x);                      // initialize stack of modules
 mextend(y->m,y->c,d ? d : q.size(),q);    // add additional module(s)
 forwardoptions(x->c, x->f, x->module());  // update options on the forward call
}

// ------------------------------------------------------------------------
// moduleget - extract module options and, optionally, parms & buffers to k
// ------------------------------------------------------------------------
static void moduleget(bool a,bool b,int64_t d,const char* s,bool t,const Module& m,K x) {
 Cast c=mcast(m); K o=moduleoptions(a,b,c,m),*k=kK(x);
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
   moduleget(a,b,d+1,i.key().c_str(),t,*i.value(),x);
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

K moduleget(bool a,bool b,const Module& m) {  
// a-true for all options else non-defaults, b-true for full state w'parms & buffers, s-name
 K k=mkeys(b),v=ktn( 0, b ? 6 : 4);  // key,val for depth,module,name,options w'parms & buffers if b
 if(container(m) || m.children().size()) {
  for(J i=0; i<v->n; ++i) kK(v)[i]=ktn(!i ? KJ : (i<3 ? KS : 0), 0);
  moduleget(a,b,0,mname(m),true,m,v);
  return xT(xD(k,v));
 } else {
  moduleget(a,b,0,mname(m),false,m,v);
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
    return moduleget(a,false,*l->m);             // return module options
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
   return kmodule(m->kmodule());
  } else if((o=xoptim(x))) {                     // optimizer ptr, extract module
   TORCH_CHECK(o->m, "module: no module registered with given optimizer");
   return kmodule(mcast(o->m),o->m);             // return new k-api handle to this module
  } else if((n=xdv(x))) {                        // depth-value pairs supplied
   return mdv(x,n);
  } else {
   return mtree(x);                              // nested tree representation
  }
 KCATCH("module");
}

// -------------------------------------------------------------------------
// modexample - given module type, return dictionary of example options
// -------------------------------------------------------------------------
K modexample(Cast c) {
 switch(c) {
  case Cast::adaptavg1d:      return knn::adapt(nn::AdaptiveAvgPool1dOptions(3));
  case Cast::adaptavg2d:      return knn::adapt(nn::AdaptiveAvgPool2dOptions({3,2}));
  case Cast::adaptavg3d:      return knn::adapt(nn::AdaptiveAvgPool3dOptions({3,2,4}));
  case Cast::adaptmax1d:      return knn::adapt(nn::AdaptiveMaxPool1dOptions(3));
  case Cast::adaptmax2d:      return knn::adapt(nn::AdaptiveMaxPool2dOptions({3,2}));
  case Cast::adaptmax3d:      return knn::adapt(nn::AdaptiveMaxPool3dOptions({3,2,4}));
  case Cast::adrop:           return knn::drop(true,nn::AlphaDropoutOptions());
  case Cast::attention:       return knn::attention(true,nn::MultiheadAttentionOptions(2048,8));
  case Cast::avgpool1d:       return knn::avgpool(true,nn::AvgPool1dOptions(3));
  case Cast::avgpool2d:       return knn::avgpool(true,nn::AvgPool2dOptions({3,2}));
  case Cast::avgpool3d:       return knn::avgpool(true,nn::AvgPool3dOptions({3,2,2}));
  case Cast::batchnorm1d:
  case Cast::batchnorm2d:
  case Cast::batchnorm3d:     return knn::batchnorm(true,nn::BatchNormOptions(32));
  case Cast::bilinear:        return knn::bilinear(true,nn::BilinearOptions(20,30,40));
  case Cast::callback:        return knn::callback(true,false,*knn::Callback(knn::CallbackOptions().fn("f").fnstring(false)));
  case Cast::cat:             return knn::dim(true,c,knn::CatOptions().dim());
  case Cast::celu:            return knn::alpha(true,nn::CELUOptions());
  case Cast::conv1d:          return knn::conv(true,nn::detail::ConvNdOptions<1>(16,32,3));
  case Cast::conv2d:          return knn::conv(true,nn::detail::ConvNdOptions<2>(16,32,{3,5}));
  case Cast::conv3d:          return knn::conv(true,nn::detail::ConvNdOptions<3>(16,32,{3,5,2}));
  case Cast::convtranspose1d: return knn::conv(true,nn::detail::ConvNdOptions<1>(128,64,5).transposed(true));
  case Cast::convtranspose2d: return knn::conv(true,nn::detail::ConvNdOptions<2>(128,64,{3,5}).transposed(true));
  case Cast::convtranspose3d: return knn::conv(true,nn::detail::ConvNdOptions<3>(128,64,{3,5,2}).transposed(true));
  case Cast::crossmap2d:      return knn::localnorm(true,c,nn::CrossMapLRN2dOptions(2));
  case Cast::decoder:         return knn::decoder(true,c, nn::TransformerDecoderOptions(
                                                  nn::TransformerDecoderLayerOptions(512,8),6)
                                                  .norm(AnyModule(nn::LayerNorm(nn::LayerNormOptions({512})))));
  case Cast::decoderlayer:    return knn::codelayer(true,c,nn::TransformerDecoderLayerOptions(512,8));
  case Cast::drop:            return knn::drop(true,nn::DropoutOptions());
  case Cast::drop2d:          return knn::drop(true,nn::Dropout2dOptions());
  case Cast::drop3d:          return knn::drop(true,nn::Dropout3dOptions());
  case Cast::droppath:        return knn::droppath(true,knn::DropPathOptions(.1));
  case Cast::elu:             return knn::alpha(true,nn::ELUOptions());
  case Cast::embed:           return knn::embed(true,c,nn::EmbeddingOptions(1000,64),{});
  case Cast::embedbag:        return knn::embed(true,c,nn::EmbeddingBagOptions(1000,64),{});
  case Cast::embedpos:        return knn::embedpos(true,knn::EmbedPositionOptions(120,512));
  case Cast::embedseq:        return knn::embedseq(true,knn::EmbedSequenceOptions(5000,512,120));
  case Cast::encoder:         return knn::encoder(true,c,nn::TransformerEncoderOptions(
                                                  nn::TransformerEncoderLayerOptions(512,8),6)
                                                  .norm(AnyModule(nn::LayerNorm(nn::LayerNormOptions({512})))));
  case Cast::encoderlayer:    return knn::codelayer(true,c,nn::TransformerEncoderLayerOptions(512,8));
  case Cast::expand:          return knn::getsize(true,knn::SizeOptions({-1,-1,28,28}));
  case Cast::fadrop:          return knn::drop(true,nn::FeatureAlphaDropoutOptions());
  case Cast::flatten:         return knn::flatten(true,nn::FlattenOptions());
  case Cast::fmaxpool2d:      return knn::fpool(true,nn::FractionalMaxPool2dOptions({2,4})  .output_size(ExpandingArray<2>({16,32})));
  case Cast::fmaxpool3d:      return knn::fpool(true,nn::FractionalMaxPool3dOptions({2,4,3}).output_size(ExpandingArray<3>({16,32,24})));
  case Cast::fold:            return knn::fold(true,nn::FoldOptions({4,6},{2,3}));
  case Cast::fork:            return KDICT;
  case Cast::gelu:            return KDICT;
  case Cast::glu:             return knn::dim(true,c,nn::GLUOptions().dim());
  case Cast::groupnorm:       return knn::groupnorm(true,nn::GroupNormOptions(3,6));
  case Cast::gru:             return knn::rnn(true,nn::GRUOptions(10,20));
  case Cast::hardshrink:      return knn::lambda(true,c,torch::nn::HardshrinkOptions().lambda());
  case Cast::hardtanh:        return knn::hardtanh(true,nn::HardtanhOptions());
  case Cast::identity:        return KDICT;
  case Cast::indexselect:     return knn::indexselect(true,knn::IndexSelectOptions(1,torch::arange(3)));
  case Cast::instancenorm1d:
  case Cast::instancenorm2d:
  case Cast::instancenorm3d:  return knn::batchnorm(true,nn::InstanceNormOptions(100));
  case Cast::interpolate:     return knn::interpolate(true,fnn::InterpolateFuncOptions().size(std::vector<int64_t>({4})));
  case Cast::layernorm:       return knn::layernorm(true,nn::LayerNormOptions({32,10}));
  case Cast::leakyrelu:       return knn::slope(true,c,nn::LeakyReLUOptions());
  case Cast::linear:          return knn::linear(true,nn::LinearOptions(784,10));
  case Cast::localnorm:       return knn::localnorm(true,c,nn::LocalResponseNormOptions(2));
  case Cast::logsigmoid:      return KDICT;
  case Cast::logsoftmax:      return knn::dim(true,c,nn::LogSoftmaxOptions(1).dim());
  case Cast::lppool1d:        return knn::lppool(true,nn::LPPool1dOptions(2,3));
  case Cast::lppool2d:        return knn::lppool(true,nn::LPPool2dOptions(1.2,{2,3}));
  case Cast::lstm:            return knn::rnn(true,nn::LSTMOptions(10,20));
  case Cast::matmul:          return KDICT;
  case Cast::maxpool1d:       return knn::maxpool(true,nn::MaxPool1dOptions(3));
  case Cast::maxpool2d:       return knn::maxpool(true,nn::MaxPool2dOptions({3,2}));
  case Cast::maxpool3d:       return knn::maxpool(true,nn::MaxPool3dOptions({3,2,2}));
  case Cast::mish:            return KDICT;
  case Cast::moduledict:      return KDICT;
  case Cast::modulelist:      return KDICT;
  case Cast::mul:             return KDICT;
  case Cast::nbeats:          return KDICT;
  case Cast::normalize:       return knn::normalize(true,fnn::NormalizeFuncOptions());
  case Cast::onehot:          return knn::onehot(true,knn::OneHotOptions(10));
  case Cast::pad:             return knn::pad(true,fnn::PadFuncOptions({1, 2, 2, 1, 1, 2}));
  case Cast::pad1d:           return knn::cpad(nn::ConstantPad1dOptions({1,2},0));
  case Cast::pad2d:           return knn::cpad(nn::ConstantPad2dOptions({1,1,2,2},0));
  case Cast::pad3d:           return knn::cpad(nn::ConstantPad3dOptions({3,3,6,6,0,1}, 3.5));
  case Cast::parmdict:        return KDICT;
  case Cast::pairwise:        return knn::pairwise(true,nn::PairwiseDistanceOptions());
  case Cast::permute:         return knn::getsize(true,knn::SizeOptions({0,2,3,1}));
  case Cast::prelu:           return knn::prelu(true,nn::PReLUOptions());
  case Cast::randomcrop:      return knn::rcrop(true,knn::RandomCropOptions(32,4).padmode(torch::kReflect));
  case Cast::randomflip:      return knn::rflip(true,knn::RandomFlipOptions(.5, -1));
  case Cast::recur:           return knn::recur(true,knn::RecurOptions());
  case Cast::reflect1d:       return knn::npad(nn::ReflectionPad1dOptions({1,2}));
  case Cast::reflect2d:       return knn::npad(nn::ReflectionPad2dOptions({1,1,2,0}));
  case Cast::relu:            return knn::inplace(true,nn::ReLUOptions().inplace());
  case Cast::relu6:           return knn::inplace(true,nn::ReLU6Options().inplace());
  case Cast::replicate1d:     return knn::npad(nn::ReplicationPad1dOptions({1,2}));
  case Cast::replicate2d:     return knn::npad(nn::ReplicationPad2dOptions({1,1,2,0}));
  case Cast::replicate3d:     return knn::npad(nn::ReplicationPad3dOptions({3,3,6,6,1,1}));
  case Cast::residual:        return KDICT;
  case Cast::reshape:         return knn::getsize(true,knn::SizeOptions({-1,1,28,28}));
  case Cast::rnn:             return knn::rnn(true,nn::RNNOptions(10,20));
  case Cast::rrelu:           return knn::rrelu(true,nn::RReLUOptions());
  case Cast::select:          return select(true,knn::SelectOptions(1,-1));
  case Cast::selfattention:   return knn::selfattn(true,knn::SelfAttentionOptions(512,8).dropout(.1));
  case Cast::selu:            return knn::inplace(true,nn::SELUOptions().inplace());
  case Cast::seqjoin:
  case Cast::seqdict:
  case Cast::seqlist:
  case Cast::seqnest:
  case Cast::sequential:      return KDICT;
  case Cast::sigmoid:         return KDICT;
  case Cast::silu:            return KDICT;
  case Cast::similar:         return knn::similar(true,nn::CosineSimilarityOptions());
  case Cast::softmax:         return knn::dim(true,c,nn::SoftmaxOptions(1).dim());
  case Cast::softmax2d:       return KDICT;
  case Cast::softmin:         return knn::dim(true,c,nn::SoftminOptions(1).dim());
  case Cast::softplus:        return knn::softplus(true,nn::SoftplusOptions());
  case Cast::softshrink:      return knn::lambda(true,c,nn::SoftshrinkOptions().lambda());
  case Cast::softsign:        return KDICT;
  case Cast::squeeze:         return knn::squeeze(true,knn::SqueezeOptions(1));
  case Cast::tanh:            return KDICT;
  case Cast::tanhshrink:      return KDICT;
  case Cast::threshold:       return knn::threshold(true,nn::ThresholdOptions(.1,0));
  case Cast::transform:       return KDICT;
  case Cast::transformer:     return knn::transformer(true,c,nn::TransformerOptions());
  case Cast::transpose:       return knn::transpose(true,knn::TransposeOptions());
  case Cast::unfold:          return knn::unfold(true,nn::UnfoldOptions({2,3}));
  case Cast::unsqueeze:       return knn::squeeze(true,knn::SqueezeOptions(0));
  case Cast::upsample:        return knn::upsample(true,nn::UpsampleOptions());
  case Cast::zeropad2d:       return knn::npad(nn::ZeroPad2dOptions({1,1,2,0}));
  case Cast::zscore:          return knn::zscore(true,
                                           knn::ZscoreOptions(torch::tensor({.51,.49,.47}).to(torch::kDouble),
                                                              torch::tensor({.25,.25,.21}).to(torch::kDouble)));
  default: TORCH_ERROR("no example options available for module enumeration: ",(I)c);
 }
}

// -----------------------------------------------------------------------
// callbacks: return table of defined callback modules w'fn, args & result
// -----------------------------------------------------------------------
KAPI kcallbacks(K x) {
 KTRY
  TORCH_CHECK(xempty(x), "callbacks: empty arg expected");
  K k=ktn(KS,3); const auto& c=env().cb; auto n=c.size(); size_t i=0;
  for(const auto& s:env().mset)
   if     (std::get<1>(s)==Setting::fn)  kS(k)[0]=std::get<0>(s);
   else if(std::get<1>(s)==Setting::in)  kS(k)[1]=std::get<0>(s);
   else if(std::get<1>(s)==Setting::out) kS(k)[2]=std::get<0>(s);
  K f=ktn(0,n), a=ktn(0,n), r=ktn(KS,n);
  for(const auto& m:c) {
   const auto& o=m->as<knn::Callback>()->options;
   kK(f)[i]=knn::cbfn(o);
   kK(a)[i]=arglist(o.in());
   kS(r)[i++]=argname(o.out());
  }
  return xT(xD(k,knk(3,f,a,r)));
 KCATCH("callbacks");
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
 fn(x, "seq",         KFN(seq),          1);    // convenience fn for sequential-like layers
 fn(x, "module",      KFN(module),       1);    // api function for module create/query
 fn(x, "adaptavg1d",  KFN(adaptavg1d),   1);    // functional form of modules/activations
 fn(x, "adaptavg2d",  KFN(adaptavg2d),   1);
 fn(x, "adaptavg3d",  KFN(adaptavg3d),   1);
 fn(x, "adaptmax1d",  KFN(adaptmax1d),   1);
 fn(x, "adaptmax2d",  KFN(adaptmax2d),   1);
 fn(x, "adaptmax3d",  KFN(adaptmax3d),   1);
 fn(x, "avgpool1d",   KFN(avgpool1d),    1);
 fn(x, "avgpool2d",   KFN(avgpool2d),    1);
 fn(x, "avgpool3d",   KFN(avgpool3d),    1);
 fn(x, "bilinear",    KFN(bilinear),     1);
 fn(x, "callbacks",   KFN(kcallbacks),   1);
 fn(x, "celu",        KFN(celu),         1);
 fn(x, "elu",         KFN(elu),          1);
 fn(x, "flatten",     KFN(flatten),      1);
 fn(x, "fmaxpool2d",  KFN(fmaxpool2d),   1);
 fn(x, "fmaxpool3d",  KFN(fmaxpool3d),   1);
 fn(x, "fold",        KFN(fold),         1);
 fn(x, "glu",         KFN(glu),          1);
 fn(x, "hardshrink",  KFN(hardshrink),   1);
 fn(x, "hardtanh",    KFN(hardtanh),     1);
 fn(x, "interpolate", KFN(interpolate),  1);
 fn(x, "leakyrelu",   KFN(leakyrelu),    1);
 fn(x, "linear",      KFN(linear),       1);
 fn(x, "logsigmoid",  KFN(logsigmoid),   1);
 fn(x, "logsoftmax",  KFN(logsoftmax),   1);
 fn(x, "lppool1d",    KFN(lppool1d),     1);
 fn(x, "lppool2d",    KFN(lppool2d),     1);
 fn(x, "maxpool1d",   KFN(maxpool1d),    1);
 fn(x, "maxpool2d",   KFN(maxpool2d),    1);
 fn(x, "maxpool3d",   KFN(maxpool3d),    1);
 fn(x, "normalize",   KFN(normalize),    1);
 fn(x, "onehot",      KFN(onehot),       1);
 fn(x, "pad",         KFN(kpad),         1);
 fn(x, "pairwise",    KFN(pairwise),     1);
 fn(x, "pdist",       KFN(pdist),        1);
 fn(x, "prelu",       KFN(prelu),        1);
 fn(x, "randomcrop",  KFN(randomcrop),   1);
 fn(x, "randomflip",  KFN(randomflip),   1);
 fn(x, "relu6",       KFN(relu6),        1);
 fn(x, "relu",        KFN(relu),         1);
 fn(x, "rrelu",       KFN(rrelu),        1);
 fn(x, "selu",        KFN(selu),         1);
 fn(x, "similar",     KFN(similar),      1);
 fn(x, "softmax",     KFN(softmax),      1);
 fn(x, "softmin",     KFN(softmin),      1);
 fn(x, "softplus",    KFN(softplus),     1);
 fn(x, "softshrink",  KFN(softshrink),   1);
 fn(x, "softsign",    KFN(softsign),     1);
 fn(x, "tanhshrink",  KFN(tanhshrink),   1);
 fn(x, "threshold",   KFN(threshold),    1);
 fn(x, "unfold",      KFN(unfold),       1);
 fn(x, "zscore",      KFN(zscore),       1);
}

/*
AdaptiveLogSoftmaxWithLoss - alternative to softmax when distribution is highly imbalanced, e.g. in language processing
normalize, interpolate  -- functional form implemented, add module?
pairwise distance & cosine similarity: in both module & functional form but forward method needs 2 input tensors
fractional pool -- try with indices registered as buffer?
embeddingbag -- forward w'defaults should work with sequential
v1.7 adds UnFlatten

adding a new module
	- if module not defined in pytorch, add .h, .cpp to knn/
	- add Cast enumeration and entry in moduleattrs(), may need to add to Settings
        - if container, need to amend container functions and addmodule to handle particular case..
        - define forward calc for requisite input arg (usually a single tensor)
	- fns to process options, e.g. options(K x,J i,Cast c) & options(bool b,const Options& o)
	- mcreate & anymodule creation or special parent creation
	- moduleoptions to return dictionary of options
	- modexample entry
*/
