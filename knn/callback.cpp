#include "../ktorch.h"
#include "callback.h"

namespace knn {

// ------------------------------------------------------------------------
// cbinput: given Pytorch tensor(s), return k container to pass to callback
// ------------------------------------------------------------------------
K cbinput(const Tensor& x) {
 return kten(x);
}

K cbinput(const Tuple& x) {
 return knk(2, kten(std::get<0>(x)), kten(std::get<1>(x)));
}

K cbinput(const Nested& x) {
 return knk(3, kten(std::get<0>(x)),
               kten(std::get<0>(std::get<1>(x))),
               kten(std::get<1>(std::get<1>(x))));
}

K cbinput(const TensorVector& x) {
 return kvec(x);
}

// ------------------------------------------------------------------------
// cbresult: get tensor/tuple/nested tuple from k callback, true if success
// ------------------------------------------------------------------------
bool cbresult(K x,Tensor& t) {
 return xten(x,t);
}

bool cbresult(K x,Tuple& t) {
 if(x->t || !x->n) {
  return false;
 } else if(xten(x,std::get<0>(t)) || (x->n==2 && xten(x,0,std::get<0>(t)) && xten(x,1,std::get<1>(t)))) {
  return true;
 } else if(auto *a=xvec(x)) {
  const auto& v=*a;
  if(v.size()>0) std::get<0>(t)=v[0];
  if(v.size()>1) std::get<1>(t)=v[1];
  if(!v.size() || v.size()>2)
   TORCH_WARN("callback: function returned a ",v.size(),"-element tensor vector, expecting 1-2 tensors for tuple");
  return true;
 } else {
  return false;
 }
}

bool cbresult(K x,Nested& n) {
 auto& t=std::get<1>(n);
 if(x->t || !x->n) {
  return false;
 } else if( xten(x,std::get<0>(n)) ||
           (x->n==2 && xten(x,0,std::get<0>(n)) && xten(x,1,std::get<0>(t))) ||
           (x->n==3 && xten(x,0,std::get<0>(n)) && xten(x,1,std::get<0>(t)) && xten(x,2,std::get<1>(t))) ) {
  return true;
 } else if(auto *a=xvec(x)) {
  const auto& v=*a;
  if(v.size()>0) std::get<0>(n)=v[0];
  if(v.size()>1) std::get<0>(t)=v[1];
  if(v.size()>2) std::get<1>(t)=v[2];
  if(!v.size() || v.size()>3)
   TORCH_WARN("callback: function returned a ",v.size(),"-element tensor vector, expecting 1-3 tensors for nested tuple");
  return true;
 } else {
  return false;
 }
}

bool cbresult(K x,TensorVector& v) {
 auto *a=xvec(x);
 return a ? v=*a,true : false;
}

// ---------------------------------------------------------------------------
// clonedict: clone parameters/buffers (used in callback clone method)
// ---------------------------------------------------------------------------
void clonedict(const TensorDict& a,TensorDict& b,const c10::optional<Device>& d) {
 for(const auto& p:a) {
  auto& t=*p;
  b[p.key()] = d && t.device() != *d ? t.to(*d) : t.clone();
 }
}

// ---------------------------------------------------------------------------
// callbacks returning tensor with 1-3 tensor inputs:
// ---------------------------------------------------------------------------
Tensor  TensorToTensorImpl::forward(const Tensor& x)                                 {return CallbackImpl::forward(x);}
Tensor Tensor2ToTensorImpl::forward(const Tensor& x,const Tensor& y)                 {return CallbackImpl::forward(x,y);}
Tensor Tensor3ToTensorImpl::forward(const Tensor& x,const Tensor& y,const Tensor& z) {return CallbackImpl::forward(x,y,z);}

AnyModule  TensorToTensorImpl::any() {return AnyModule(std::dynamic_pointer_cast<TensorToTensorImpl>(shared_from_this()));}
AnyModule Tensor2ToTensorImpl::any() {return AnyModule(std::dynamic_pointer_cast<Tensor2ToTensorImpl>(shared_from_this()));}
AnyModule Tensor3ToTensorImpl::any() {return AnyModule(std::dynamic_pointer_cast<Tensor3ToTensorImpl>(shared_from_this()));}

Moduleptr  TensorToTensorImpl::clone(const c10::optional<Device>& d) const {return CallbackImpl::clone<TensorToTensorImpl>(d);}
Moduleptr Tensor2ToTensorImpl::clone(const c10::optional<Device>& d) const {return CallbackImpl::clone<Tensor2ToTensorImpl>(d);}
Moduleptr Tensor3ToTensorImpl::clone(const c10::optional<Device>& d) const {return CallbackImpl::clone<Tensor3ToTensorImpl>(d);}

// -----------------------------------------------------------------------------
// cbfn: set/get callback fn as symbol or string, e.g. `f or "{[m;x] mul(x;x)}"
// -----------------------------------------------------------------------------
static void cbfn(S s,K x,CallbackOptions& o) {
 if(s) {
  TORCH_CHECK(!nullsym(s), "callback function cannot be null");
  o.fn(s);
  o.fnstring(false);
 } else {
  TORCH_CHECK(x->n, "callback function cannot be an empty string");
  std::string s;
  s.assign((S)kC(x),x->n);
  o.fn(s);
  o.fnstring(true);
 }
}

static void cbfn(K x,J i,CallbackOptions& o) {
 S s;
 if(xsym(x,i,s)) {
  cbfn(s,nullptr,o);
 } else if(!x->t && kK(x)[i]->t==KC) {
  cbfn(nullptr,kK(x)[i],o);
 } else {
  TORCH_ERROR("callback function expected as symbol or string, given ",kname(x,i));
 }
}

static void cbfn(const Pairs& p,CallbackOptions& o) {
 if(p.t==-KS) {
  cbfn(p.s,nullptr,o);
 } else if (p.t==KC) {
  cbfn(nullptr,p.v,o);
 } else {
  TORCH_ERROR("callback function expected as symbol or string, given ",kname(p.t));
 }
}

K cbfn(const CallbackOptions& o) {
 return o.fnstring() ? kp((S)o.fn().c_str()) : ks(cs(o.fn().c_str()));
}

// -----------------------------------------------------------------------------
// cbin:  map from input(s), e.g. `tensor`tuple -> Arg::tensor,Arg::tuple
// cbout: map from result type symbol to enumeration, `tensor -> Arg::tensor
// -----------------------------------------------------------------------------
static Args cbin(K x) {
 SymArrayRef s;
 TORCH_CHECK(!xempty(x), "callback in: not defined, given empty list");
 TORCH_CHECK(xsyms(x,s), "callback in: expecting symbol(s), given ",kname(x));
 Args v;
 for(auto a:s) v.push_back(argtype(a,"input"));
 return v;
}

static Args cbin(K x,J i) {
 S s;
 return xsym(x,i,s) ? Args{argtype(s,"input")} : cbin(kK(x)[i]);
}

static Args cbin(const Pairs& p) {
 TORCH_CHECK(p.t==-KS || p.t==KS, "callback ",p.k,": expected symbol(s), given ",kname(p.t));
 return p.t==-KS ? Args{argtype(p.s,"input")} : cbin(p.v);
}

static Arg cbout(K x,J i) {
 S r;
 TORCH_CHECK(xsym(x,i,r), "callback out: expected symbol, given ",kname(x,i));
 return argtype(r,"result");
}

static Arg cbout(const Pairs& p) {
 TORCH_CHECK(p.t==-KS, "callback ",p.k,": expected symbol, given ",kname(p.t));
 return argtype(p.s,"result");
}

// --------------------------------------------------------------------
// cbpair: true if final arg should be evaulated for name-value pair(s)
// --------------------------------------------------------------------
static bool cbpair(K x) {
 if(x->t || x->n<2)
  return false;
 K y=kK(x)[x->n-1];
 if(xdict(y)) y=kK(y)[0];
 if(y->t<0 || !xlen(y))
  return false;
 S k;
 if(!xsyms((!y->t && y->n) ? kK(y)[0] : y, k))
  return false;
 auto s=Setting::undefined;
 for(const auto& a:env().mset) {
  if(std::get<0>(a)==k) {
   s=std::get<1>(a); break;
  }
 }
 switch(s) {
  case Setting::fn:
  case Setting::in:
  case Setting::out:
  case Setting::parms:
  case Setting::buffers: 
   return true;
  default:
   return false;
 }
}

// ---------------------------------------------------------------------------
// cbparms: set callback parameters & buffers
// ---------------------------------------------------------------------------
static void cbparms(K x,J i,Setting s,TensorDict& d) {
 TORCH_CHECK(!x->t, "callback ",mset(s),": unexpected ",kname(x));
 auto *a=xtensordict(x,i);
 d=a ? *a : kputd(kK(x)[i]);
}

static void cbparms(const Pairs& p,TensorDict& d) {
 TORCH_CHECK(!p.t || p.t==99, "callback ",p.k,": expecting k dictionary or (syms;vals), given ",kname(p.t));
 auto *a=xtensordict(p.v);
 d=a ? *a : kputd(p.v);
}

// ---------------------------------------------------------------------------
// cbclone: create specific callback matching arg(s) & result
// ---------------------------------------------------------------------------
static Moduleptr cbclone(const CallbackOptions& o) {
 for(const auto& a:env().cb) {
  if(a->as<Callback>()->options.out() == o.out() && a->as<Callback>()->options.in() == o.in()) {
   auto m=a->clone();
    m->as<Callback>()->options.fn(o.fn()).fnstring(o.fnstring());
   return m;
  }
 }
 std::string s("in=");
 for(auto a:o.in()) {s+=argname(a); s+=",";}
 TORCH_ERROR("callback: no module defined for ",s," out=",argname(o.out()),")");
}

// ---------------------------------------------------------------------------
// callback: setting & retrieving options for callback modules
// ---------------------------------------------------------------------------
Moduleptr callback(K x,J i,Cast c) {
 CallbackOptions o; Pairs p; TensorDict pm,bf;
 J n=cbpair(x) ? xargc(x,i,p) : xlen(x)-i;
 for(J j=0;j<n;++j)
   switch(j) {
    case 0: cbfn(x,i+j,o); break;
    case 1: o.in(cbin(x,i+j)); break;
    case 2: o.out(cbout(x,i+j)); break;
    case 3: cbparms(x,i+j,Setting::parms,pm); break;
    case 4: cbparms(x,i+j,Setting::buffers,bf); break;
    default: mpos(x,c,i+j); break;
   }
 while(xpair(p))
  switch(mset(p.k,c)) {
   case Setting::fn:      cbfn(p,o); break;
   case Setting::in:      o.in(cbin(p)); break;
   case Setting::out:     o.out(cbout(p)); break;
   case Setting::parms:   cbparms(p,pm); break;
   case Setting::buffers: cbparms(p,bf); break;
   default: mpair(c,p); break;
  }
 if(!o.fn().size()) {
  if(x->t==KS && x->n==2) {
   TORCH_ERROR(msym(c),": no k function defined, 2nd symbol, `",kS(x)[1],", defines callback module name");
  } else {
   TORCH_ERROR(msym(c),": no k function defined");
  }
 }
 TORCH_CHECK(o.in().size(), msym(c),": no input type(s) given");
 auto m=cbclone(o);
 for(const auto& t:pm) m->register_parameter(t.key(), t.value());
 for(const auto& t:bf) m->register_buffer   (t.key(), t.value());
 return m;
}

K callback(bool a,bool b,const CallbackImpl& m) {
 K x=KDICT; auto o=m.options; decltype(o) d;
 msetting(x, Setting::fn, cbfn(o));
 if(a || o.in()  != d.in())  msetting(x, Setting::in, arglist(o.in()));
 if(a || o.out() != d.out()) msetting(x, Setting::out, ks(argname(o.out())));
 if(!b) {
  const auto& p=m.named_parameters(false);
  const auto& f=m.named_buffers(false);
  if(a || p.size()) msetting(x, Setting::parms,   kget(p));
  if(a || f.size()) msetting(x, Setting::buffers, kget(f));
 }
 return resolvedict(x);
}

// ---------------------------------------------------------------------------
// callbacks - list of possible callbacks based on result type & input arg(s)
// ---------------------------------------------------------------------------
Callbacks callbacks() {
 auto o=CallbackOptions().fnstring(true).out(Arg::tensor);
 return {{
   TensorToTensor(o.fn("{[m;x]}").in({Arg::tensor})).ptr(),
  Tensor2ToTensor(o.fn("{[m;x;y]}").in({Arg::tensor,Arg::tensor})).ptr(),
  Tensor3ToTensor(o.fn("{[m;x;y;z]}").in({Arg::tensor,Arg::tensor,Arg::tensor})).ptr()
 }};
}

} // namespace knn
