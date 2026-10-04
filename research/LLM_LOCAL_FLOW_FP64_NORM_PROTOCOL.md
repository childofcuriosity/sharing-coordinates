# Dtype-preserving normalization follow-up

2026-09-25, after the FP32 run and while the first FP64 precision follow-up is running. Local source inspection found that Hugging Face LlamaRMSNorm.forward explicitly casts its input to float32 even after model.double(). The first FP64 follow-up therefore retains internal FP32 normalization; preserve it and label it accordingly.

Run the SAME four rates, state, real text batch, SGD objective and horizon on another GPU. Additionally replace RMSNorm forward with the identical normalization formula evaluated in the input dtype (FP64): variance=mean(x*x); output=weight*x*rsqrt(variance+epsilon). Preserve epsilon and all pretrained weights. Assert double input at each replaced normalization and shared injection. Record the count of replaced normalization modules. SDPA receives FP64 activations; fixed rotary positional constants may still originate from FP32, but do not depend on trained factors.

This changes numerical precision, not the mathematical normalization, task, initialization or selected example. Retain the original FP32 and outer-FP64 traces; do not claim they demonstrated first-order convergence if they did not. This is a precision diagnostic selected after numerical issues, not a new independent replication or an independent natural-training sample.
