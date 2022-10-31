#include "../ktorch.h"
#include "transform.h"

namespace knn {

ZscoreImpl::ZscoreImpl(const ZscoreOptions& o) : options(o) {reset();}

void ZscoreImpl::reset() {
 TORCH_CHECK(options.mean().defined()   && options.mean().is_floating_point(),   "zscore: mean(s) not defined as float/double");
 TORCH_CHECK(options.stddev().defined() && options.stddev().is_floating_point(), "zscore: std deviation(s) not defined as float/double");
 mean=this->register_buffer("mean", options.mean());
 stddev=this->register_buffer("stddev", options.stddev());
}

void ZscoreImpl::pretty_print(std::ostream& s) const {
  s << std::boolalpha;
  s << "knn::Zscore(mean="; print_tensor(s,3,mean);
  s <<         ", stddev="; print_tensor(s,3,stddev);
  s <<         ", inplace=" << options.inplace() << ")";
 }

Tensor ZscoreImpl::forward(Tensor t) {
 return options.inplace() ? zscore_(t,mean,stddev) : zscore(t,mean,stddev);
}

// ---------------------------------------------------------------------------
// zscore - set/get mean,stddev & inplace flag for zscore module
// ---------------------------------------------------------------------------
ZscoreOptions zscore(K x,J i,Cast c) {
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

K zscore(bool a,const ZscoreOptions& o) {
 K x=KDICT;
 msetting(x, Setting::mean, kget(o.mean()));
 msetting(x, Setting::std,  kget(o.stddev()));
 if(a || o.inplace()) msetting(x, Setting::inplace, kb(o.inplace()));
 return resolvedict(x);
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

// ------------------------------------------------------------------
// random crop - pad, then crop tensor height,width at random offsets
// ------------------------------------------------------------------
RandomCropImpl::RandomCropImpl(torch::ExpandingArray<2> s,torch::ExpandingArray<4> p) : RandomCropImpl(RandomCropOptions(s,p)) {}
RandomCropImpl::RandomCropImpl(const RandomCropOptions& o) : options(o) {reset();}

void RandomCropImpl::reset() {}

void RandomCropImpl::pretty_print(std::ostream& s) const {
  s << "knn::RandomCrop(size=" << options.size();
  if(*options.pad() != *torch::ExpandingArray<4>(0)) s << ", pad=" << options.pad();
  s << ")";
 }

Tensor RandomCropImpl::forward(const Tensor& t) {
 return rcrop(t,options);
}

// ---------------------------------------------------------------------------
// padmode - match k symbol to std::variant style enumeration
// rcrop - set/get probability p and dim for random crop
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

RandomCropOptions rcrop(K x,J i,Cast c) {
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

K rcrop(bool a,const RandomCropOptions& o) {
 K x=KDICT; const RandomCropOptions d(0);
 msetting(x, Setting::size, KEX(o.size()));
 if(a || *d.pad()            != *o.pad())            msetting(x, Setting::pad,     KEX(o.pad()));
 if(a || d.padmode().index() != o.padmode().index()) msetting(x, Setting::padmode, ks(ESYM(o.padmode())));
 if(a || d.value()           != o.value())           msetting(x, Setting::value,   kf(o.value()));
 return resolvedict(x);
}

// ---------------------------------------------------------------------------
// cpad - handle padding prior to random cropping
// rcrop - perform random crop given size & padding options
// ---------------------------------------------------------------------------
static Tensor cpad(const Tensor& t,const RandomCropOptions& o) {
 return *o.pad() == *ExpandingArray<4>(0) ? t : torch::nn::functional::detail::pad(t,o.pad(),o.padmode(),o.value());
}

Tensor rcrop(const Tensor& t,const RandomCropOptions& o) {
 using namespace torch::indexing;
 auto d=t.dim(); auto h=(*o.size())[0]; auto w=(*o.size())[1];
 TORCH_CHECK(1<d && d<5, "randomcrop: not implemented for ",d,"-dim tensor");
 auto n=d<4 ? 1 : t.size(0);
 auto p=cpad(t,o); auto r=p.size(-2); auto c=p.size(-1);
 if(r==h && c==w) {                      // if crop size matches tensor rows & cols
  return p;                              // return tensor as is
 } else {
  auto y=r-h+1; auto x=c-w+1;            // else set possible range for top left corner
  TORCH_CHECK(x>0 && y>0, "crop: size ",h,"x",w,", exceeds tensor dim(s) ",r,"x",c);
  auto ry=torch::randint(y,n,torch::kLong); auto *py=ry.data_ptr<int64_t>();
  auto rx=torch::randint(x,n,torch::kLong); auto *px=rx.data_ptr<int64_t>();
  if(n<=1) {
   y=py[0]; x=px[0];
   return p.index({Ellipsis, Slice(y,y+h), Slice(x,x+w)});
  } else {
   auto r=torch::empty_like(t);
   for(const auto i:c10::irange(n)) {
    y=py[i]; x=px[i];
    r[i]=p[i].index({Ellipsis, Slice(y,y+h), Slice(x,x+w)});
   }
   return r;
  }
 }
}

// ------------------------------------------------------------------
// random flip, horizontal or vertical depending on dimension given
// ------------------------------------------------------------------
RandomFlipImpl::RandomFlipImpl(double p,int64_t d) : RandomFlipImpl(RandomFlipOptions(p,d)) {}
RandomFlipImpl::RandomFlipImpl(const RandomFlipOptions& o) : options(o) {reset();}

void RandomFlipImpl::reset() {}

void RandomFlipImpl::pretty_print(std::ostream& s) const {
 s << "knn::RandomFlip(p=" << options.p() << ", dim=" << options.dim() << ")";
}

Tensor RandomFlipImpl::forward(const Tensor& t) {
 return rflip(t,options);
}

// ---------------------------------------------------------------------------
// rflip - set/get probability p and dim for random horizontal/vertical flip
//       - also perform the flip given input, options w'prob and dimension
// ---------------------------------------------------------------------------
RandomFlipOptions rflip(K x,J i,Cast c) {
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

K rflip(bool a,const RandomFlipOptions& o) {
 K x=KDICT; const RandomFlipOptions d;
 if(a || d.p()   != o.p())   msetting(x, Setting::p,   kf(o.p()));
 if(a || d.dim() != o.dim()) msetting(x, Setting::dim, kj(o.dim()));
 return resolvedict(x);
}

Tensor rflip(const Tensor& t,const RandomFlipOptions& o) {
 auto d=t.dim();
 TORCH_CHECK(1<d && d<5, "randomflip: not implemented for ",d,"-dim tensor");
 auto n=d<4 ? 1 : t.size(0);
 auto i=torch::nonzero(torch::rand(n,torch::kDouble) < o.p()).flatten();
 if(i.numel()) {
  if(n==1) {
   return t.flip(o.dim());
  } else {
   auto *p=i.data_ptr<int64_t>();
   auto r=t.clone();
   for(const auto j:c10::irange(i.numel())) {
    auto k=p[j];
    r[k]=t[k].flip(o.dim());
   }
   return r;
  }
 } else {
  return t;
 }
}

// -------------------------------------------------------------------------------
// transform: define sequential modules to transform data in training & eval mode
// -------------------------------------------------------------------------------
void TransformImpl::reset() {}
void TransformImpl::pretty_print(std::ostream& s) const {s << "knn::Transform(";}

void TransformImpl::push_back(const torch::nn::Sequential& q) {
 push_back(children().size()==0 ? "train" : "eval", q);
}

void TransformImpl::push_back(std::string s,const torch::nn::Sequential& q) {
 TORCH_CHECK(children().size()<2, "transform: both training and evaluation transforms already defined");
 if(children().size()==0)
  train=register_module(s,std::move(q));
 else
  eval=register_module(s,std::move(q));
}

void TransformImpl::push_back(const torch::nn::AnyModule& a) {
 TORCH_CHECK(false, "transform: cannot add ",mlabel(a.ptr())," module, expecting sequential block");
}

void TransformImpl::push_back(std::string s,const torch::nn::AnyModule& a) {
 TORCH_CHECK(false, "transform: cannot add ",mlabel(a.ptr())," module, expecting sequential block");
}

Tensor TransformImpl::forward(const Tensor& x) {
 if(is_training()) {
  return train.is_empty() || !train->children().size() ? x : train->forward(x);
 } else {
  return eval.is_empty()  || !eval->children().size()  ? x : eval->forward(x);
 }
}

} // namespace knn
