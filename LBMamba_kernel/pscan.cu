#include <torch/extension.h>


std::vector<at::Tensor>
selective_scan_fwd(const at::Tensor & u,
                   const at::Tensor & delta,
                   const at::Tensor & A,
                   const at::Tensor & B,
                   const at::Tensor & C,
                   const c10::optional<at::Tensor> & D_,
                   const c10::optional<at::Tensor> & z_,
                   const c10::optional<at::Tensor> & delta_bias_,
                   bool delta_softplus,
                   bool out_float = true);


std::vector<at::Tensor>
selective_scan_bwd(const at::Tensor & u,
                   const at::Tensor & delta,
                   const at::Tensor & A,
                   const at::Tensor & B,
                   const at::Tensor & C,
                   const c10::optional<at::Tensor> & D_,
                   const c10::optional<at::Tensor> & z_,
                   const c10::optional<at::Tensor> & delta_bias_,
                   const at::Tensor & dout,
                   const c10::optional<at::Tensor> & x_,
                   const c10::optional<at::Tensor> & out_,
                   c10::optional<at::Tensor> & dz_,
                   bool delta_softplus,
                   bool recompute_out_z);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    using namespace pybind11::literals;

    m.def("fwd",
          &selective_scan_fwd,
          "Selective scan forward",
          "u"_a,
          "delta"_a,
          "A"_a,
          "B"_a,
          "C"_a,
          "D_"_a,
          "z_"_a,
          "delta_bias_"_a,
          "delta_softplus"_a,
          "out_float"_a = true);

    m.def("bwd",
          &selective_scan_bwd,
          "u"_a,
          "delta"_a,
          "A"_a,
          "B"_a,
          "C"_a,
          "D_"_a,
          "z_"_a,
          "delta_bias_"_a,
          "dout"_a,
          "x_"_a,
          "out_"_a,
          "dz_"_a,
          "delta_softplus"_a,
          "recompute_out_z"_a);
}
