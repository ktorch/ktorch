#pragma once
#include "util.h"

namespace knn {

// -----------------------------------------------------------------------------
// padmode - translate symbol to/from variant used for padding mode
// padsym - translate symbol to padding for same or valid padding 
// convpad - translate input(symbol or long(s)) into padding for convolution
// conv - create 1-3d convolution, set dictionary given module
//        w'version 1.4, c++ ConvImpl class was split into regular & transposed
//        ConvOptions & ConvTransOptions have different members, 
// convtran - similar to conv() but adds output_padding, changes position order
// -----------------------------------------------------------------------------
torch::nn::detail::conv_padding_mode_t padmode(S);
S padmode(const torch::nn::detail::conv_padding_mode_t&);


template<size_t D> torch::nn::detail::conv_padding_t<D> padsym(S s,Cast c) {
 switch(emap(s)) {
  case Enum::same:  return torch::kSame;
  case Enum::valid: return torch::kValid;
  default: TORCH_ERROR(msym(c),": unrecognized padding: ",s); break;
 }
}

template<size_t D> torch::nn::detail::conv_padding_t<D> convpad(K x,J i,Cast c) {
 S s; return xsym(x,i,s) ? padsym<D>(s,c) : exarray<D>(x,i,c,Setting::pad);
}

template<size_t D> torch::nn::detail::conv_padding_t<D> convpad(const Pairs& p,Cast c) {
 return p.t == -KS ? padsym<D>(p.s,c) : exarray<D>(p,c);
}

template<size_t D> torch::nn::ConvOptions<D> conv(K x,J i,Cast c) {
 torch::nn::ConvOptions<D> o(0,0,0);
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

template<size_t D> torch::nn::ConvTransposeOptions<D> convtran(K x,J i,Cast c) {
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

template<size_t D> void convpad(K x,bool a,const torch::nn::detail::conv_padding_t<D>& d,const torch::nn::detail::conv_padding_t<D>& o) {
 if(auto p=std::get_if<ExpandingArray<D>>(&o)) {
  auto pd=std::get_if<ExpandingArray<D>>(&d);
  if(a || !pd || **pd != **p)
   msetting(x, Setting::pad, KEX((*p)));
 } else if(a || d.index() != o.index()) {
  if(std::get_if<torch::enumtype::kSame>(&o)) {
   msetting(x, Setting::pad, ks(emap(Enum::same)));
  } else if(std::get_if<torch::enumtype::kValid>(&o)) {
   msetting(x, Setting::pad, ks(emap(Enum::valid)));
  } else {
   TORCH_ERROR("unrecognized convolution padding");
  }
 }
}

template<size_t D> K conv(bool a,const torch::nn::detail::ConvNdOptions<D>& o) {
 K x=KDICT; torch::nn::detail::ConvNdOptions<D> d(o.in_channels(),o.out_channels(),o.kernel_size());
 bool t=o.transposed();
 msetting(x, Setting::in,   kj(o.in_channels()));
 msetting(x, Setting::out,  kj(o.out_channels()));
 msetting(x, Setting::size, KEX(o.kernel_size()));
 if(a || (*o.stride()  != *d.stride()))  msetting(x, Setting::stride, KEX(o.stride()));
 convpad<D>(x,a,d.padding(),o.padding());
 if(t) {
  if(a || (*o.output_padding() != *d.output_padding())) msetting(x, Setting::outpad, KEX(o.output_padding()));
 } else {
  if(a || (*o.dilation() != *d.dilation())) msetting(x, Setting::dilate, KEX(o.dilation()));
 }
 if(a || ( o.groups()    !=  d.groups())) msetting(x, Setting::groups, kj(o.groups()));
 if(a || ( o.bias()      !=  d.bias()))   msetting(x, Setting::bias,   kb(o.bias()));
 if(t) {
  if(a || (*o.dilation() != *d.dilation())) msetting(x, Setting::dilate, KEX(o.dilation()));
 }
 if(a || o.padding_mode().index() != d.padding_mode().index()) msetting(x, Setting::padmode, ks(padmode(o.padding_mode())));
 return resolvedict(x);
}

} // namespace knn
