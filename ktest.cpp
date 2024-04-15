#include "ktorch.h"

#ifdef __clang__
# pragma clang diagnostic push
# pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"    // FORWARD_HAS_DEFAULT_ARGS
#endif

#include "torch/script.h"
#include "knn.h"
namespace nn=torch::nn;
namespace fnn=torch::nn::functional;

//#include "c10d/NCCLUtils.hpp"
//#include "torch/csrc/cuda/nccl.h"

using ParmMap=torch::OrderedDict<std::string, std::vector<Tensor>>;  // no. of gradients x no. of workers, e.g. 31 x 2

ParmMap buildmap(std::vector<Moduleptr> x) {
 ParmMap m;
 for(const auto& a:x) {
  for(auto& p:a->named_parameters()) {
   if(auto *v=m.find(p.key()))
    v->push_back(p.value());
   else
    m.insert(p.key(), {p.value()});
  }
 }
 return m;
}

ParmMap buildmap(const Module& l) {
 ParmMap m;
 for(const auto& a:l.children()) {
  for(auto& p:a->named_parameters()) {
   if(auto *v=m.find(p.key()))
    v->push_back(p.value());
   else
    m.insert(p.key(), {p.value()});
  }
 }
 return m;
}

static void adatest() {
 auto x=torch::tensor({0.5, 2.0, 4.0}, torch::requires_grad());
 auto y=torch::tensor({1.0, 2.0, 3.0});
 //auto o=torch::optim::SGD(std::vector<torch::Tensor>{}, torch::optim::SGDOptions(0.1));
 auto o=torch::optim::Adagrad(std::vector<torch::Tensor>{}, torch::optim::AdagradOptions(0.1));
 auto& p=o.param_groups();
 p[0].params().push_back(x);
 auto l=torch::nn::functional::mse_loss(x,y);
 l.backward();
 o.step();
}

KAPI Adatest(K x) {
 KTRY
  adatest();
  return (K)0;
 KCATCH("AdaGrad test");
}

KAPI matches(K x,K y) {
 KTRY
  Tensor a,b;
  TORCH_CHECK(xten(x,a), "1st arg not a tensor");
  TORCH_CHECK(xten(y,b), "2nd arg not a tensor");
  return kten(torch::stack({a.eq(b).sum(), torch::tensor(a.numel())}));
 KCATCH("matches");
}

KAPI gen(K x) {
 KTRY
  Device d(DeviceType::CPU);
  if(xempty(x)) {
   TORCH_ERROR("nyi");
  } else if(xdev(x,d)) {
   //auto g=torch::make_generator<torch::CPUGeneratorImpl>();
   const auto& c1=torch::globalContext().defaultGenerator(d).clone();
   std::cerr << "default generator with seed: " << (J)c1.current_seed() << "\n";
   auto c2=torch::globalContext().defaultGenerator(d).clone();
   std::cerr << " cloned generator with seed: " << (J)c2.current_seed() << "\n";
   const auto& g=torch::globalContext().defaultGenerator(d);
   std::cerr << "generator defined: " << g.defined() << "\n";
   //std::cerr << "generator seed: " << g.seed() << "\n";
   return kten(g.get_state());
  } else {
   TORCH_ERROR("nyi");
  }
 KCATCH("gen");
}

KAPI nccl(K x) {
 KTRY
  Tensor a,b;
  TORCH_CHECK(xten(x,0,a), "1st arg must be a tensor");
  TORCH_CHECK(xten(x,1,b), "2nd arg must be a tensor");
  //return kb(torch::cuda::nccl::is_available({a,b}));
  return (K)0;
 KCATCH("nccl");
}

KAPI gradtest(K x,K y) {
 KTRY
  Tensor a;
  TORCH_CHECK(xten(x,a),"1st arg of tensor expected");
  Tensor b=kput(y);
  auto& g=a.mutable_grad();
  std::cerr<< "gradient defined:" << g.defined() << "\n";
  return (K)0;
 KCATCH("gradtest");
}

KAPI gradcopy(K x) {
 KTRY
  TORCH_CHECK(!x->t,"gradcopy: expecting general list, given ",kname(x));
  TORCH_CHECK(x->n==3,"gradcopy: expecting 3 args, (dict;name;values), ",x->n," given");
  auto *d=xtensordict(x,0); S s;
  TORCH_CHECK(d, "gradcopy: expecting 1st arg of a dictionary of parameters, given ",kname(x,0));
  TORCH_CHECK(xsym(x,1,s), "gradcopy: expecting 2nd arg of a parameter name, given ",kname(x,1));
  auto *t=d->find(s);
  TORCH_CHECK(t, "gradcopy: unable to find parameter `",s);
  auto& g=t->mutable_grad();
  TORCH_CHECK(g.defined(),"gradcopy: parameter `",s," gradient not defined");
  g.copy_(kput(x,2).view(g.sizes()));
  return (K)0;
 KCATCH("gradcopy");
}

static bool xfn(K x,std::string& s) {
 bool b=true;
 if(xsym(x))
  s=x->s;
 else if(x->t==KC)
  s.assign((S)kC(x),x->n);
 else
  b=false;
 return b;
}

static bool xfn(K x,J i,std::string& s) {return xind(x,i) && xfn(kK(x)[i],s);}

static bool xint(K x,I& i) {
 bool b=true;
 switch(x->t) {
  case -KI: i=x->i; break;
  case -KJ: i=x->j; break;
  default: b=false; break;
 }
 return b;
}

static bool xint(K x,J j,I& i) {return xind(x,j) && xint(kK(x)[j],i);}

static unsigned gradhook(const Tensor& t,int h,S f,S s) {
 return t.register_hook([h,f,s](const Tensor& x) {
  K r;
  if(h) {
   r=k(0, (S)"{[h;f;s;x] h(f;s;x)}", ki(h), ks(f), ks(s), kget(x.flatten()), (K)0);
   //std::cerr << h << ", " << s << ",nan: " << x.isnan().any() << ", inf: " << x.isinf().any() << ", size " << x.sizes() << ", fn: " << f << "\n";
/*
   K a=ks(s),z=kget(x.flatten());
   r=k(h,f,a,z,(K)0);
*/
   //r=k(h,f,ks(s),kget(x.flatten()),(K)0);
   //r=k(h,f,ks(s),kget(x.flatten().zero_()),(K)0);
  } else {
   K g=kten(x);
   r=k(0,f,ks(s),r1(g),(K)0);
   TORCH_CHECK(xfree(g),"hook: unable to free gradient for parameter `",s); r0(g);
  }
  TORCH_CHECK(r, "hook: network error, gradient callback for parameter `",s);
  if(h>=0) {
   if(r->t == -128) {
    std::string e(r->s); r0(r);
    TORCH_ERROR("hook: gradient callback for parameter `",s,", ",e);
   } else {
    r0(r);
   }
  }
 }); 
}

static K gradhook(const TensorDict& d,int h,S f) {
 size_t i=0; K k=ktn(KS,d.size()), v=ktn(KJ,d.size());
 for(const auto& a:d.items()) {
  S s=cs(a.key().c_str()); kS(k)[i]=s; kJ(v)[i++]=gradhook(a.value(),h,f,s);
 }
 return xD(k,v);
}

// unhook   void remove_hook(unsigned pos) const;
KAPI hook1(K x) {
 KTRY
  TORCH_CHECK(!x->t,"hook: not implemented for arg of ",kname(x));
  TORCH_CHECK(x->n==3, "hook: expecting 3 args, (object;handle;callback), given ",x->n);
  auto *m=xmodel(x,0); auto *q=m ? m->kmodule() : xmodule(x,0); auto *d=q ? nullptr : xtensordict(x,0);
  TORCH_CHECK(q || d, "hook: expecting 1st arg of model, module or parameter dictionary, given ",kname(x,0));
  I h; std::string f;
  TORCH_CHECK(xint(x,1,h), "hook: expecting 2nd arg of k connection handle (long or integer), given ",kname(x,1));
  TORCH_CHECK(xfn(x,2,f), "hook: expecting 3rd arg of callback function name (symbol) or expression (string), given ",kname(x,2));
  TORCH_CHECK(f.size(), "hook: callback function name/expression is empty");
  //const auto& p=q ? q->module().named_parameters() : *d;
  return (K)0; //return gradhook(p,h,f);
 KCATCH("hook");
}

// unhook   void remove_hook(unsigned pos) const;
KAPI hook(K x) {
 KTRY
  TORCH_CHECK(!x->t,"hook: not implemented for arg of ",kname(x));
  TORCH_CHECK(x->n==3, "hook: expecting 3 args, (object;handle;callback), given ",x->n);
  auto *m=xmodel(x,0); auto *q=m ? m->kmodule() : xmodule(x,0); auto *d=q ? nullptr : xtensordict(x,0);
  TORCH_CHECK(q || d, "hook: expecting 1st arg of model, module or parameter dictionary, given ",kname(x,0));
  I h; S f;
  TORCH_CHECK(xint(x,1,h), "hook: expecting 2nd arg of k connection handle (long or integer), given ",kname(x,1));
  TORCH_CHECK(xsym(x,2,f), "hook: expecting 3rd arg of callback function name (symbol), given ",kname(x,2));
  //TORCH_CHECK(f.size(), "hook: callback function name/expression is empty");
  const auto& p=q ? q->module().named_parameters() : *d;
  return gradhook(p,h,f);
 KCATCH("hook");
}

KAPI gradsend(K x) {
 KTRY
  TORCH_CHECK(!x->t,"gradsend: not implemented for arg of ",kname(x));
  TORCH_CHECK(x->n==3, "gradsend: expecting 3 args, (object;handle;callback), given ",x->n);
  auto *m=xmodel(x,0); auto *q=m ? m->kmodule() : xmodule(x,0); auto *d=q ? nullptr : xtensordict(x,0);
  TORCH_CHECK(q || d, "gradsend: expecting 1st arg of model, module or parameter dictionary, given ",kname(x,0));
  I h; S f;
  TORCH_CHECK(xint(x,1,h), "gradsend: expecting 2nd arg of k connection handle (long or integer), given ",kname(x,1));
  TORCH_CHECK(xsym(x,2,f), "gradsend: expecting 3rd arg of callback function name (symbol), given ",kname(x,2));
  const auto& p=q ? q->module().named_parameters() : *d;
  for(const auto& a:p.items()) {
   const auto& g=a.value().grad();
   if(g.defined()) {
    S s=cs(a.key().c_str());
    K r=k(h,f,ks(s),kget(g.flatten()),(K)0);
    TORCH_CHECK(r, "gradsend: network error, gradient callback for parameter `",s);
    if(h>=0) {
     if(r->t == -128) {
      std::string e(r->s); r0(r);
      TORCH_ERROR("gradsend: parameter `",s,", ",e);
     } else {
      r0(r);
     }
    }
   }
  }
  return (K)0;
 KCATCH("gradsend");
}

KAPI testhalf(K x) {
 KTRY
  auto t=torch::tensor({1.0,2.0,3.0});
  std::cerr << t << "\n";
  t=t.cuda();
  std::cerr << t << "\n";
  t=t.to(torch::kFloat16);
  std::cerr << t << "\n";
  t=t.to(torch::kBFloat16);
  std::cerr << t << "\n";
  return (K)0;
 KCATCH("testhalf");
}

// -------------------------------------------------------------------
// ewa
// -------------------------------------------------------------------
static void ewacalc(const TensorDict& src,const TensorDict& tgt,double b,int64_t n,const char *c) {
 auto a=1.0 - b;
 if(n>0) {   // apply bias correction factor
  auto f=1.0 / (1-std::pow(b,n)); a*=f;
 }
 for(const auto& x:src) {
  auto *y=tgt.find(x.key()); 
  TORCH_CHECK(y, "ewa: unable to find target ",c," `",x.key());
  TORCH_CHECK(x.value().sizes()==y->sizes(), "ewa: size mismatch, ",c," `",x.key(),", source size ",x.value().sizes()," vs target ",y->sizes());
  const auto& v=x.value().to(y->device());
  if(v.is_floating_point()) {
   y->mul_(b).add_(v,a);
  } else {     // integral value: assume buffer, e.g. counts, copy directly
   y->copy_(v);
  }
 }
}

KAPI ewa(K x) {
 KTRY
  torch::NoGradGuard g; double b; int64_t n=0;
  TORCH_CHECK(!x->t, "ewa: not implemented for ",kname(x));
  TORCH_CHECK(x->n==3 || x->n==4, "ewa: expecting 3-4 args, (source;target;beta;step), given ",x->n," arg(s)");
  auto *m=xmodel(x,0); auto *q=m ? m->kmodule() : xmodule(x,0);
  TORCH_CHECK(q, "ewa: 1st arg of source module or model expected, given ",kname(x,0));
  const auto& src=q->module();
  m=xmodel(x,1); q=m ? m->kmodule() : xmodule(x,1);
  TORCH_CHECK(q, "ewa: 2nd arg of target module or model expected, given ",kname(x,1));
  auto& tgt=q->module();
  TORCH_CHECK(xdouble(x,2,b), "ewa: 3rd arg of beta (the exponential moving average factor) is expected as double, given ",kname(x,0));
  TORCH_CHECK(0<b && b<1.0, "ewa: beta is expected to be positive and less than 1, given ",b);
  TORCH_CHECK(x->n==3 || xint64(x,3,n), "ewa: 4th arg of step is expected as long, given ",kname(x,3));
  TORCH_CHECK(n>=0, "ewa: step cannot be negative, given ",n);
  ewacalc(src.named_parameters(), tgt.named_parameters(), b, n, "parameter");
  ewacalc(src.named_buffers(),    tgt.named_buffers(),    b, n, "buffer");
  return (K)0;
 KCATCH("ewa");
}

KAPI whiten(K x) {
 KTRY
  Tensor e,t,v;
  bool b=xten(x,t); if(!b) t=kput(x);
  int64_t h=3, w=3, c=t.size(1);
  auto p=t.unfold(2,h,1).unfold(3,w,1);
  p=p.transpose(1,3).reshape({-1,c,h,w}).to(torch::kFloat);
  const auto& s=p.sizes();
  auto n=s[0]; c=s[1],h=s[2],w=s[3];
  auto X=p.reshape({n,c*h*w});
  X=X/std::sqrt(X.size(0)-1);
  std::tie(e,v)=torch::linalg_eigh(X.t().mm(X));
  e=e.flip(0);
  v=v.t().reshape({c*h*w,c,h,w}).flip(0);
  return kresult(b, (v / torch::sqrt(e + 1e-2).view({-1,1,1,1})));
 KCATCH("whiten");
}

KAPI mixup(K a) {
 KTRY
  Tensor x,y; xtenarg(a,x,y);
  double a=1.0;
  int64_t n=10;
  auto l=torch::_sample_dirichlet(torch::tensor({a,a}))[0].item<double>();
  TensorVector v;
  x=x.is_floating_point() ? x.clone() : x.to(torch::kFloat);
  v.emplace_back(x.clone().mul_(l).add_(x.roll(1,0).mul(1-l)));
  y=y.dim()>1 ? y.clone() : torch::nn::functional::one_hot(y,n).to(x.is_floating_point() ? x.scalar_type() : torch::kFloat);
  v.emplace_back(y.mul_(l).add_(y.roll(1,0).mul(1-l)));
  return kvec(v);
 KCATCH("mixup");
}

static const std::vector<torch::indexing::TensorIndex> cutindex(int64_t h,int64_t w,double& l) {
 using namespace torch::indexing;
 auto r=0.5 * std::sqrt(1-l);
 auto rw = (int64_t)floor(r*w);
 auto rh = (int64_t)floor(r*h);
 auto rx=torch::randint(w,1).item<int64_t>();
 auto ry=torch::randint(h,1).item<int64_t>();
 auto x1=rx-rw; if(x1<0) x1=0; auto x2=rx+rw; if(x2>w) x2=w;
 auto y1=ry-rh; if(y1<0) y1=0; auto y2=ry+rh; if(y2>h) y2=h;
 l=(x2-x1) * (y2-y1);
 l=1.0 - l/(w * h);
 return {Ellipsis, Slice(y1,y2), Slice(x1,x2)};
}

KAPI cutmix(K a) {
 KTRY
  Tensor x,y; xtenarg(a,x,y);
  double a=1.0;
  int64_t n=10;
  auto l=torch::_sample_dirichlet(torch::tensor({a,a}))[0].item<double>();
  auto i=cutindex(x.size(-2),x.size(-1),l);
  TensorVector v;
  v.emplace_back(x.clone().index_put_(i, x.roll(1,0).index(i)));
  y=y.dim()>1 ? y.clone() : torch::nn::functional::one_hot(y,n).to(x.is_floating_point() ? x.scalar_type() : torch::kFloat);
  v.emplace_back(y.mul_(l).add_(y.roll(1,0).mul(1-l)));
  return kvec(v);
 KCATCH("cutmix");
}

KAPI cutmix1(K x) {
 KTRY
  Tensor t;
  if(!xten(x,t)) t=kput(x);
  auto h=t.size(-2);
  auto w=t.size(-1);
  double a=1.0;
  auto l=torch::_sample_dirichlet(torch::tensor({a,a}))[0].item<double>();
  auto r=0.5 * std::sqrt(1-l);
  auto rw = (int64_t)floor(r*w);
  auto rh = (int64_t)floor(r*h);
  auto rx=torch::randint(w,1).item<int64_t>();
  auto ry=torch::randint(h,1).item<int64_t>();
  auto x1=rx-rw; if(x1<0) x1=0;
  auto y1=ry-rh; if(y1<0) y1=0;
  auto x2=rx+rw; if(x2>w) x2=w;
  auto y2=ry+rh; if(y2>h) y2=h;
  l=(x2-x1) * (y2-y1);
  l=1.0 - l/(w * h);
  std::cerr << "lambda: " << l << "\n";
  return kget(torch::tensor({y1,y2,x1,x2},torch::kLong));
 KCATCH("cutmix");
}

Tensor blend(const Tensor& x,const Tensor& y,double r) {
 return (r*x + (1.0-r)*y).clamp(0.0,x.is_floating_point() ? 1.0 : 255.0).to(x.dtype());
}

Tensor brightness(const Tensor& x,double f) {
 TORCH_CHECK(f>=0, "brightness: factor must be non-negative, given ",f);
 return (f*x).clamp(0.0,x.is_floating_point() ? 1.0 : 255.0).to(x.dtype());
}

/*
def rand_brightness(x):
    x = x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
    return x
*/

Tensor grayscale(const Tensor& x) {
 return(0.2989 * x.select(-3,0) + 0.587 * x.select(-3,1) + 0.114 * x.select(-3,2)).to(x.dtype()).unsqueeze(-3);
 // r.expand(x.size())
 // contrast,saturation
}

Tensor contrast(const Tensor& x,double f) {
 TORCH_CHECK(f>=0, "contrast: factor must be non-negative, given ",f);
 auto c=x.size(-3);
 auto d=x.is_floating_point() ? x.scalar_type() : torch::kFloat;
 return blend(x, (c==3 ? grayscale(x) : x).mean({-3,-2,-1}, true, d), f);
}

Tensor saturation(const Tensor& x,double f) {
 TORCH_CHECK(f>=0, "saturation: factor must be non-negative, given ",f);
 // if 1 channel, return x
 return blend(x, grayscale(x), f);
}

/*
def rand_saturation(x):
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + x_mean
    return x
*/

Tensor imageinvert(const Tensor& x) {
 return torch::tensor(x.is_floating_point() ? 1 : 255, TensorOptions().dtype(x.dtype()).device(x.device())) - x;
}

Tensor solarize(const Tensor& x,double f) {
 auto b=x.is_floating_point() ? 1.0 : 255.0;
 TORCH_CHECK(f<=b, "solarize: threshold should be less than ",b,", given ",f);
 return torch::where(x >= f, imageinvert(x), x);
}

Tensor blur(const Tensor& x) {
 auto d=x.is_floating_point() ? x.scalar_type() : torch::kFloat;
 auto k=torch::tensor({1,1,1,1,5,1,1,1,1}, TensorOptions().dtype(d).device(x.device()));
 k/=k.sum();
 k=k.reshape({3,3}).expand({x.size(-3),1,3,3});
 auto c=torch::nn::functional::conv2d(x.to(d),k,torch::nn::functional::Conv2dFuncOptions().groups(x.size(-3)));
 // convert c to x type
 auto r=x.clone();
 return c;
}

//https://github.com/lucidrains/lightweight-gan/blob/main/lightweight_gan/diff_augment.py

/*

    result_tmp, need_cast, need_squeeze, out_dtype = _cast_squeeze_in( img, [kernel.dtype,],)
    result_tmp = conv2d(result_tmp, kernel, groups=result_tmp.shape[-3])
    result_tmp = _cast_squeeze_out(result_tmp, need_cast, need_squeeze, out_dtype)

    result = img.clone()
    result[..., 1:-1, 1:-1] = result_tmp

    return result


def adjust_sharpness(img: Tensor, sharpness_factor: float) -> Tensor:
    if sharpness_factor < 0:
        raise ValueError(f"sharpness_factor ({sharpness_factor}) is not non-negative.")

    _assert_image_tensor(img)

    _assert_channels(img, [1, 3])

    if img.size(-1) <= 2 or img.size(-2) <= 2:
        return img

    return _blend(img, _blurred_degenerate_image(img), sharpness_factor)

*/

KAPI rtest(K x) {
 KTRY
  Tensor t;
  TORCH_CHECK(xten(x,t), "need tensor");
  auto u=torch::randint(3,10,torch::kLong);
  int64_t *r=u.data_ptr<int64_t>();
  std::cerr << u << "\n";
  for(const auto i:c10::irange(1000000)) {
   auto y=t+r[1]+i;
  }
  return (K)0;
 KCATCH("rtest");
}

KAPI amask(K x) {
 KTRY
  TORCH_CHECK(x->t == -KJ, "mask: need long scalar for mask size, given ",kname(x));
  TORCH_CHECK(x->j >= 0, "mask: size must be non-negative");
  // check IEEE754 support here since -inf is not guaranteed to be valid on non IEEE754 platform
  if (std::numeric_limits<float>::is_iec559) {
    return kten(torch::triu(torch::full({x->j, x->j}, -std::numeric_limits<float>::infinity()), 1));
  } else { // if IEEE754 is not supported, we use the smallest float number in current platform
    TORCH_WARN_ONCE("IEEE754 is not supported, mask will use smallest float number on this platform instead of -inf");
    return kten(torch::triu(torch::full({x->j, x->j}, std::numeric_limits<float>::lowest()), 1));
  }
 KCATCH("mask");
}

KAPI sizes_strides(K x,K y) {
 KTRY
  Tensor *t=xten(x);
  TORCH_CHECK(t,"supply tensor");
  IntArrayRef sz;
  TORCH_CHECK(xsize(y,sz), "supply size");
  t->unsafeGetTensorImpl()->set_sizes_and_strides(sz,{});
  return (K)0;
 KCATCH("sizes & strides");
}

/*
KAPI ganstep(K a) {
 KTRY
  Kmodel *d=xmodel(a,0), *g=xmodel(a,1);
  TORCH_CHECK(d && g, "ganstep: supply discriminator & generator model as 1st & 2nd args");
  TORCH_CHECK(d->o.c != Cast::lbfgs, "Cannout use lbfgs optimizer with discriminator");
  TORCH_CHECK(g->o.c != Cast::lbfgs, "Cannout use lbfgs optimizer with generator");
  Tensor* x=xten(a,1); Tensor* y=xten(a,2); Tensor* z=xten(a,3);
  d->opt().zero_grad();
  Tensor l0=mloss(d, *x, (*y)[0]);
  l0.backward();
  Tensor gx=c10::get<Tensor>(mforward(g->kmodule(),*z));
  Tensor l1=mloss(d, gx.detach(), (*y)[1]);
  l1.backward();
  d->opt().step();
  g->opt().zero_grad();
  Tensor l2=mloss(d, gx, (*y)[2]);
  l2.backward();
  g->opt().step();
  return(K)0;
 KCATCH("ganstep");
}
*/

J sbytes(const Tensor& t,std::unordered_set<intptr_t>& u) {
 if(t.use_count()>1 || t.storage().use_count()>1) { // multiple references
  auto p=(intptr_t)t.storage().data();              // get integer pointer
  if(u.count(p)) {                                  // if seen before
   return 0;                                        // don't count bytes
  } else {                                          // else
   u.emplace(p);                                    // add pointer to set
   return t.storage().nbytes();                     // return bytes allocated
  }
 } else {
  return t.storage().nbytes();                      // no multiple references
 }
}

template<typename V>J sbytes(const V& v) {
  J n=0; std::unordered_set<intptr_t> u;
  for(const auto& t:v)
   if(t.defined())
    n += sbytes(t,u);
  return n;
}

J dbytes(const TensorDict& d) {
  J n=0; std::unordered_set<intptr_t> u;
  for(const auto& a:d)
   if(a.value().defined())
    n += sbytes(a.value(),u);
  return n;
}

//J vecbytes(const TensorVector& v) {
template<typename V>J vecbytes(const V& v) {
  J n=0; std::unordered_set<intptr_t> s;
  for(size_t i=0; i<v.size(); ++i) {
   if(v[i].storage().use_count()>1) {       // more than 1 tensor uses the storage
    auto p=(intptr_t)v[i].storage().data(); // get integer pointer
    if(!s.count(p)) {                       // if not seen before
     n += v[i].storage().nbytes();          // add the bytes allocated
     s.emplace(p);                          // add pointer to set
    }
   } else {
    n += v[i].storage().nbytes();
   }
  }
  return n;
}

KAPI bytes2(K x) {
 KTRY
  Kmodule *m=xmodule(x);
  auto *v=xvec(x);
  auto *d=xtensordict(x);
  if(m)
   return kj(sbytes(m->m->parameters()));
  else if(v)
   return kj(sbytes(*v));
  else if(d)
   return kj(dbytes(*d));
  else
   TORCH_ERROR("not module/vector");
 KCATCH("bytes2");
}

KAPI contigflag(K x) {
 KTRY
  Attr a=Attr::contiguous; K y=nullptr;
  Ktag *g=xtag(x);
  if(!g) {
   g=xtag(x,0);
   TORCH_CHECK(!g || x->n==2, mapattr(a),": expecting up to 2 args, given ",x->n);
   y=kK(x)[1];
  }
  TORCH_CHECK(g, mapattr(a),": expecting object, e.g. tensor, vector, module");
  TORCH_CHECK(g->a==Class::tensor,": not a tensor");
  TORCH_CHECK(!y || y->t==-KS, mapattr(a),": additional arg not a symbol");
  const Tensor& t=g->tensor();
  return kb(y ? t.is_contiguous(optmemory(y->s)) : t.is_contiguous());
 KCATCH("contiguous");
}

KAPI tf32(K x) {
 KTRY
 TORCH_CHECK(x->t==-KB,"need boolean scalar");
  std::cerr << " cuBLAS: " << torch::globalContext().allowTF32CuBLAS() << "\n";
  std::cerr << " cuDNN:  " << torch::globalContext().allowTF32CuDNN() << "\n";
  torch::globalContext().setAllowTF32CuBLAS(x->g);
  torch::globalContext().setAllowTF32CuDNN(x->g);
  std::cerr << " cuBLAS: " << torch::globalContext().allowTF32CuBLAS() << "\n";
  std::cerr << " cuDNN:  " << torch::globalContext().allowTF32CuDNN() << "\n";
  return (K)0;
 KCATCH("tf32");
}

KAPI coo(K x) {
 auto o=torch::dtype(torch::kDouble).device(torch::kCUDA);
 auto t1=torch::sparse_coo_tensor({2,3}, o);
 std::cerr << t1 << "\n";

 auto i=torch::tensor({{1},{1}});
 auto v=torch::tensor({1});
 auto t2=torch::sparse_coo_tensor(i,v,o);
 std::cerr << t2 << "\n";

 return (K)0;
}

KAPI sdim(K x,K y) {
 KTRY
  Tensor *t=xten(x);
  TORCH_CHECK(t, "tensor for 1st arg");
  TORCH_CHECK(y->t==-KJ, "long dim for 2nd arg");
  //return kten(torch::native::dense_to_sparse(*t,y->j));
  return kten(t->to_sparse(y->j));
 KCATCH("test dense_to_sparse");
}

KAPI optparse(K x) {
 KTRY
  TensorOptions o;
  TORCH_CHECK(xopt(x,o), "need tensor options");
  return optmap(o);
 KCATCH("optparse");
}

/*
d=torch.nn.ParameterDict()
for k,v in model.named_parameters():
  d[k.replace('.','_')]=v
 
torch.jit.save(torch.jit.script(d),"/tmp/gpt")
*/

KAPI jfile(K x) {
 KTRY
  TORCH_CHECK(x->t==-KS, "need symbol");
  torch::jit::script::Module j = torch::jit::load(x->s);
  TensorDict d;
  for(const auto& a:j.named_parameters())
   d.insert(a.name,a.value);
  return kdict(d);
 KCATCH("load file");
}

std::vector<char> getbytes(std::string f) {
    std::ifstream s(f, std::ios::binary);
    std::vector<char> v((std::istreambuf_iterator<char>(s)), (std::istreambuf_iterator<char>()));
    s.close();
    return v;
}

KAPI loadfile(K x) {
 KTRY
  TORCH_CHECK(x->t==-KS, "need symbol");
  auto v=getbytes(x->s);
  std::cerr << "read file: " << x->s << ", " << v.size() << " byte(s)\n";
  torch::IValue a = torch::pickle_load(v);
  //  torch::Tensor my_tensor = x.toTensor();
  return (K)0;
 KCATCH("load file");
}

/*
//#include  <c10/cuda/CUDAUtils.h>
KAPI dtest(K x) {
 auto d=c10::cuda::current_device();
 std::cerr << d << "\n";
 return(K)0;
}
*/

/*
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
*/

//template <class... F> struct overload : F... {overload(F... f) : F(f)... {}};
//template <class... F> auto make_overload(F... f) {return overload<F...>(f...);}

/*
void testover(auto& x,auto& y) {
 auto f=make_overload(
   [](J& x, J& y) {std::cerr << x+y <<"\n"},
   [](auto& x, auto& y) {std::cerr << "auto\n"}),
   x,y);
}
*/

KAPI ksizes(K x) {
 std::cerr << "type_info: " << sizeof(std::type_info) << "\n";
 J j; auto h=typeid(j).hash_code();
 std::vector<TensorVector> v(3);
 std::cerr << "hash_code: " << sizeof(h) << "\n";
 std::cerr << "k0:      " << sizeof(k0) << "\n";
 std::cerr << "Tensor:  " << sizeof(Tensor) << "\n";
 std::cerr << "Module:  " << sizeof(Module) << "\n";
 std::cerr << "Class:   " << sizeof(Class) << "\n";
 std::cerr << "Cast:    " << sizeof(Cast) << "\n";
 std::cerr << "Ktag:    " << sizeof(Ktag) << "\n";
 std::cerr << "Kten:    " << sizeof(Kten) << "\n";
 std::cerr << "Kvec:    " << sizeof(Kvec) << "\n";
 std::cerr << "Kdict:   " << sizeof(Kdict) << "\n";
 std::cerr << "Kmodule: " << sizeof(Kmodule) << "\n";
 std::cerr << "Kopt:    " << sizeof(Kopt) << "\n";
 std::cerr << "Kmodel:  " << sizeof(Kmodel) << "\n";
 std::cerr << "Tuple:   " << sizeof(Tuple) << "\n";
 std::cerr << "Moduleptr: " << sizeof(Moduleptr) << "\n";
 std::cerr << "TensorVector: " << sizeof(TensorVector) << "\n";
 std::cerr << "TensorDict: " << sizeof(TensorDict) << "\n";
 std::cerr << "Input:  " << sizeof(Input) << "\n";
 std::cerr << "Output:  " << sizeof(Output) << "\n";
 std::cerr << "optional bool:  " << sizeof(c10::optional<bool>) << "\n";
 std::cerr << "optional int64: " << sizeof(c10::optional<int64_t>) << "\n";
 std::cerr << "Training options: " << sizeof(TrainOptions) << "\n";
 std::cerr << "Small TensorVector(5): " << sizeof(torch::SmallVector<Tensor,5>) << "\n";
 std::cerr << "Small TensorVector(7): " << sizeof(torch::SmallVector<Tensor,7>) << "\n";
 std::cerr << "Small TensorVector(9): " << sizeof(torch::SmallVector<Tensor,9>) << "\n";
 std::cerr << "vector of vectors: " << sizeof(v) << "\n";
 std::cerr << "size of Data: " << sizeof(Data) << "\n";
 std::cerr << "size of optional Data: " << sizeof(c10::optional<Data>) << "\n";
 return (K)0;
}

KAPI randint_type(K x) {
 std::cerr << torch::randint(10,{3}) << "\n";
 return (K)0;
}

//#include <c10/cuda/CUDAMacros.h>
//#include <c10/cuda/CUDACachingAllocator.h>

// check for cuda via USE_CUDA
// #ifdef USE_CUDA
//  ..
// #endif
/*
namespace c10 {
namespace cuda {
namespace CUDACachingAllocator {
C10_CUDA_API void emptyCache();
C10_CUDA_API uint64_t currentMemoryAllocated(int device);
C10_CUDA_API uint64_t maxMemoryAllocated(int device);
C10_CUDA_API void     resetMaxMemoryAllocated(int device);
C10_CUDA_API uint64_t currentMemoryCached(int device);
C10_CUDA_API uint64_t maxMemoryCached(int device);
C10_CUDA_API void     resetMaxMemoryCached(int device);
}}}
*/

/*
cache      
memory     e.g. memory() or memory`cuda or memory 0
maxcache   
maxmemory  
emptycache
resetcache 
resetmemory
*/
KAPI cudamem(K x) {
 KTRY
  // if sym, get device no
  // if int, verify -1<n< env.cuda
  //auto n=c10::cuda::CUDACachingAllocator::currentMemoryAllocated(x->j);
  //return kj(n);
  return kj(nj);
 KCATCH("cuda memory");
}

#define ENUMTEST(name) \
{ \
  v = torch::k##name; \
  std::cerr << torch::enumtype::get_enum_name(v) << " " << ESYM(v) << "\n"; \
}

KAPI enumtest(K x) {
  std::variant<
    torch::enumtype::kLinear,
    torch::enumtype::kConv1D,
    torch::enumtype::kConv2D,
    torch::enumtype::kConv3D,
    torch::enumtype::kConvTranspose1D,
    torch::enumtype::kConvTranspose2D,
    torch::enumtype::kConvTranspose3D,
    torch::enumtype::kSigmoid,
    torch::enumtype::kTanh,
    torch::enumtype::kReLU,
    torch::enumtype::kLeakyReLU,
    torch::enumtype::kFanIn,
    torch::enumtype::kFanOut,
    torch::enumtype::kGELU,
    torch::enumtype::kConstant,
    torch::enumtype::kReflect,
    torch::enumtype::kReplicate,
    torch::enumtype::kCircular,
    torch::enumtype::kNearest,
    torch::enumtype::kNearestExact,
    torch::enumtype::kBilinear,
    torch::enumtype::kBicubic,
    torch::enumtype::kTrilinear,
    torch::enumtype::kArea,
    torch::enumtype::kSum,
    torch::enumtype::kMean,
    torch::enumtype::kMax,
    torch::enumtype::kNone,
    torch::enumtype::kBatchMean,
    torch::enumtype::kZeros,
    torch::enumtype::kBorder,
    torch::enumtype::kReflection,
    torch::enumtype::kMish,
    torch::enumtype::kSame,
    torch::enumtype::kSiLU,
    torch::enumtype::kValid
  > v;

  ENUMTEST(Linear)
  ENUMTEST(Conv1D)
  ENUMTEST(Conv2D)
  ENUMTEST(Conv3D)
  ENUMTEST(ConvTranspose1D)
  ENUMTEST(ConvTranspose2D)
  ENUMTEST(ConvTranspose3D)
  ENUMTEST(Sigmoid)
  ENUMTEST(Tanh)
  ENUMTEST(ReLU)
  ENUMTEST(LeakyReLU)
  ENUMTEST(FanIn)
  ENUMTEST(FanOut)
  ENUMTEST(GELU)
  ENUMTEST(Constant)
  ENUMTEST(Reflect)
  ENUMTEST(Replicate)
  ENUMTEST(Circular)
  ENUMTEST(Nearest)
  ENUMTEST(NearestExact)
  ENUMTEST(Bilinear)
  ENUMTEST(Bicubic)
  ENUMTEST(Trilinear)
  ENUMTEST(Area)
  ENUMTEST(Sum)
  ENUMTEST(Mean)
  ENUMTEST(Max)
  ENUMTEST(None)
  ENUMTEST(BatchMean)
  ENUMTEST(Zeros)
  ENUMTEST(Border)
  ENUMTEST(Reflection)
  ENUMTEST(Mish)
  ENUMTEST(Same)
  ENUMTEST(SiLU)
  ENUMTEST(Valid)
 return (K)0;
}

KAPI kdata(K x,K y) {
 KTRY
  int64_t i=0;
  auto dataset = torch::data::datasets::MNIST(x->s)
    .map(torch::data::transforms::Normalize<>(0.5, 0.5))
    .map(torch::data::transforms::Stack<>());
  auto data_loader = torch::data::make_data_loader(std::move(dataset));
  for (torch::data::Example<>& batch : *data_loader) {
    if(i==y->j) {
     std::cout << batch.target << "\n";
     std::cout << batch.data   << "\n ";
     return kten(batch.data);
    }
  }
  return (K)0;
 KCATCH("mnist test");
}

#ifdef __clang__
# pragma clang diagnostic pop
#endif
