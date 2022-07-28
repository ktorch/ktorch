#pragma once
#include "util.h"

namespace knn {

// --------------------------------------------------------------------------------------
// fold,unfold - set/get size,dilation,padding,stride
// --------------------------------------------------------------------------------------
torch::nn::FoldOptions fold(K,J,Cast);
K fold(bool,const torch::nn::FoldOptions&);

torch::nn::UnfoldOptions unfold(K,J,Cast);
K unfold(bool,const torch::nn::UnfoldOptions&);

} // knn namespace
