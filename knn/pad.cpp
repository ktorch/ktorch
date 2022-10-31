#include "../ktorch.h"
#include "pad.h"

namespace knn {

PadImpl::PadImpl(std::vector<int64_t> p) : PadImpl(torch::nn::functional::PadFuncOptions(p)) {}
PadImpl::PadImpl(const torch::nn::functional::PadFuncOptions& o) : options(o) {reset();}

Tensor PadImpl::forward(const Tensor& x) {
 return torch::nn::functional::pad(x,options);
}

void PadImpl::reset() {}

void PadImpl::pretty_print(std::ostream& s) const {
 s << "knn::Pad(" << "pad=" << options.pad()
   << ", mode=" << torch::enumtype::get_enum_name(options.mode())
   << ", value=" << options.value() << ")";
}

// ----------------------------------------------------------------------------------
// padmode - match k symbol to std::variant style enumeration
// pad - n-dimensional padding, specify even number of sizes and optional pad value
//       given options, returns k dictionary of module settings
// ----------------------------------------------------------------------------------
static void padmode(torch::nn::functional::PadFuncOptions& o,S s) {
 switch(emap(s)) {
  case Enum::constant:  o.mode(torch::kConstant); break;
  case Enum::reflect:   o.mode(torch::kReflect); break;
  case Enum::replicate: o.mode(torch::kReplicate); break;
  case Enum::circular:  o.mode(torch::kCircular); break;
  default: TORCH_ERROR("unrecognized padding mode: ",s); break;
 }
}

torch::nn::functional::PadFuncOptions pad(K x,J i,Cast c) {
 torch::nn::functional::PadFuncOptions o({}); S s; Pairs p; J n=xargc(x,i,p);
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

K pad(bool a,const torch::nn::functional::PadFuncOptions& o) {
 K x=KDICT; const torch::nn::functional::PadFuncOptions d({});
 msetting(x, Setting::pad, klist(o.pad().size(),o.pad().data()));
 if(a || o.mode().index() != d.mode().index()) msetting(x, Setting::mode,  ks(ESYM(o.mode())));
 if(a || o.value()        != d.value())        msetting(x, Setting::value, kf(o.value()));
 return resolvedict(x);
}

} // namespace knn

