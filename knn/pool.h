#pragma once

namespace knn {

// ----------------------------------------------------------------------------------
//  maxpool - process args, return dictionary of options, call functional form
// ----------------------------------------------------------------------------------
template<size_t D> torch::nn::MaxPoolOptions<D> maxpool(K x,J i,Cast c) {
 torch::nn::MaxPoolOptions<D> o(0);
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

template<typename O> K maxpool(bool a,const O& o) {
 K x=KDICT; O d(o.kernel_size());
 msetting(x, Setting::size, KEX(o.kernel_size()));
 if(a || *o.stride()   != *d.stride())   msetting(x, Setting::stride,  KEX(o.stride()));
 if(a || *o.padding()  != *d.padding())  msetting(x, Setting::pad,     KEX(o.padding()));
 if(a || *o.dilation() != *d.dilation()) msetting(x, Setting::dilate,  KEX(o.dilation()));
 if(a || o.ceil_mode() != d.ceil_mode()) msetting(x, Setting::ceiling, kb(o.ceil_mode()));
 return x;
}

// ----------------------------------------------------------------------------------
//  avgpool - process args, return dictionary of options, call functional form
// ----------------------------------------------------------------------------------
template<size_t D> torch::nn::AvgPoolOptions<D> avgpool(K x,J i,Cast c) {
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

template<typename O> K avgpool(bool a,const O& o) {
 K x=KDICT; O d(o.kernel_size());
 msetting(x, Setting::size, KEX(o.kernel_size()));
 if(a || *o.stride()           != *d.stride())           msetting(x, Setting::stride,   KEX(o.stride()));
 if(a || *o.padding()          != *d.padding())          msetting(x, Setting::pad,      KEX(o.padding()));
 if(a || o.ceil_mode()         != d.ceil_mode())         msetting(x, Setting::ceiling,  kb(o.ceil_mode()));
 if(a || o.count_include_pad() != d.count_include_pad()) msetting(x, Setting::countpad, kb(o.count_include_pad()));
 if(a || o.divisor_override().has_value())               msetting(x, Setting::divisor,  kj(o.divisor_override() ? o.divisor_override().value() : nj));
 return x;
}

// ---------------------------------------------------------------------------------------
// adaptive pooling - process args, return dictionary of options, call functional form
// adapt - multiple versions to handle expanding array(1d) vs array of optionals(2,3d)
// ---------------------------------------------------------------------------------------
template<size_t D> void adapt(ExpandingArray<D>& a,K x,J i,Cast c)        {a=exarray<D>(x,i,c,Setting::size);}
template<size_t D> void adapt(ExpandingArray<D>& a,const Pairs& p,Cast c) {a=exarray<D>(p,c);}
template<size_t D> void adapt(Exoptional<D>& a,K x,J i,Cast c)        {a=exoptional<D>(x,i,c,Setting::size);}
template<size_t D> void adapt(Exoptional<D>& a,const Pairs& p,Cast c) {a=exoptional<D>(p,c);}

template<size_t D> bool adapt(ExpandingArray<D>& a) {for(const auto &v:*a) if(v != nj) return true; return false;}
template<size_t D> bool adapt(Exoptional<D>& a)     {for(const auto &v:*a) if(v)       return true; return false;}

template<size_t D,typename T> T adapt(K x,J i,Cast c) {
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

template<typename O> K adapt(const O& o) {
 K x=KDICT;
 msetting(x, Setting::size, KEX(o.output_size()));
 return x;
}

// ----------------------------------------------------------------------------------
// fpool - fractional max pooling for 2 & 3d layers
// ----------------------------------------------------------------------------------
template<size_t D> torch::nn::FractionalMaxPoolOptions<D> fpool(K x,J i,Cast c) {
 torch::nn::FractionalMaxPoolOptions<D> o(0);
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

template<typename O> K fpool(bool a,const O& o) {
 K x=KDICT;
 msetting(x, Setting::size, KEX(o.kernel_size()));
 if(a || o.output_size().has_value())    msetting(x, Setting::outsize, o.output_size() ? KEX(o.output_size().value())  : ktn(0,0));
 if(a || o.output_ratio().has_value())   msetting(x, Setting::ratio,   o.output_ratio()? KEX(o.output_ratio().value()) : ktn(0,0));
 return x;
}

// ----------------------------------------------------------------------------------
// lppool - power-average pooling
// ----------------------------------------------------------------------------------
template<size_t D> torch::nn::LPPoolOptions<D> lppool(K x,J i,Cast c) {
 torch::nn::LPPoolOptions<D> o(0,0);
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

template<typename O> K lppool(bool a,const O& o) {
 K x=KDICT; O d(o.norm_type(),o.kernel_size());
 msetting(x, Setting::p,    kf(o.norm_type()));
 msetting(x, Setting::size, KEX(o.kernel_size()));
 if(a || *o.stride()   != *d.stride())   msetting(x, Setting::stride,  KEX(o.stride()));
 if(a || o.ceil_mode() != d.ceil_mode()) msetting(x, Setting::ceiling, kb(o.ceil_mode()));
 return x;
}
} // knn namespace
