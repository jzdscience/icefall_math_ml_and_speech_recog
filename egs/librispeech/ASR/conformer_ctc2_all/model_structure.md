Conformer(
  (encoder_embed): Conv2dSubsampling(
    (conv): Sequential(
      (0): ScaledConv2d(1, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
      (1): ActivationBalancer()
      (2): DoubleSwish()
      (3): ScaledConv2d(8, 32, kernel_size=(3, 3), stride=(2, 2))
      (4): ActivationBalancer()
      (5): DoubleSwish()
      (6): ScaledConv2d(32, 128, kernel_size=(3, 3), stride=(2, 2))
      (7): ActivationBalancer()
      (8): DoubleSwish()
    )
    (out): ScaledLinear(in_features=2432, out_features=512, bias=True)
    (out_norm): BasicNorm()
    (out_balancer): ActivationBalancer()
  )
  (encoder_pos): RelPositionalEncoding(
    (dropout): Dropout(p=0.1, inplace=False)
  )
  (encoder): ConformerEncoder(
    (layers): ModuleList(
      (0-11): 12 x ConformerEncoderLayer(
        (self_attn): RelPositionMultiheadAttention(
          (in_proj): ScaledLinear(in_features=512, out_features=1536, bias=True)
          (out_proj): ScaledLinear(in_features=512, out_features=512, bias=True)
          (linear_pos): ScaledLinear(in_features=512, out_features=512, bias=False)
        )
        (feed_forward): Sequential(
          (0): ScaledLinear(in_features=512, out_features=2048, bias=True)
          (1): ActivationBalancer()
          (2): Swish()
          (3): Dropout(p=0.1, inplace=False)
          (4): ScaledLinear(in_features=2048, out_features=512, bias=True)
        )
        (feed_forward_macaron): Sequential(
          (0): ScaledLinear(in_features=512, out_features=2048, bias=True)
          (1): ActivationBalancer()
          (2): Swish()
          (3): Dropout(p=0.1, inplace=False)
          (4): ScaledLinear(in_features=2048, out_features=512, bias=True)
        )
        (conv_module): ConvolutionModule(
          (pointwise_conv1): ScaledConv1d(512, 1024, kernel_size=(1,), stride=(1,))
          (deriv_balancer1): ActivationBalancer()
          (depthwise_conv): ScaledConv1d(512, 512, kernel_size=(31,), stride=(1,), padding=(15,), groups=512)
          (deriv_balancer2): ActivationBalancer()
          (activation): Swish()
          (pointwise_conv2): ScaledConv1d(512, 512, kernel_size=(1,), stride=(1,))
        )
        (norm_final): BasicNorm()
        (balancer): ActivationBalancer()
        (dropout): Dropout(p=0.1, inplace=False)
      )
    )
  )
  (encoder_output_layer): Sequential(
    (0): Dropout(p=0.1, inplace=False)
    (1): ScaledLinear(in_features=512, out_features=500, bias=True)
  )
  (decoder_embed): ScaledEmbedding(500, 512)
  (decoder_pos): PositionalEncoding(
    (dropout): Dropout(p=0.1, inplace=False)
  )
  (decoder): TransformerDecoder(
    (layers): ModuleList(
      (0-5): 6 x TransformerDecoderLayer(
        (self_attn): MultiheadAttention(
          (in_proj_weight): ScaledLinear(in_features=512, out_features=1536, bias=True)
          (out_proj): ScaledLinear(in_features=512, out_features=512, bias=True)
        )
        (src_attn): MultiheadAttention(
          (in_proj_weight): ScaledLinear(in_features=512, out_features=1536, bias=True)
          (out_proj): ScaledLinear(in_features=512, out_features=512, bias=True)
        )
        (feed_forward): Sequential(
          (0): ScaledLinear(in_features=512, out_features=2048, bias=True)
          (1): ActivationBalancer()
          (2): DoubleSwish()
          (3): Dropout(p=0.1, inplace=False)
          (4): ScaledLinear(in_features=2048, out_features=512, bias=True)
        )
        (norm_final): BasicNorm()
        (balancer): ActivationBalancer()
        (dropout): Dropout(p=0.1, inplace=False)
      )
    )
  )
  (decoder_output_layer): ScaledLinear(in_features=512, out_features=500, bias=True)
  (decoder_criterion): LabelSmoothingLoss()
)