# Peaked-circuits
    Over a decade after its original proposal, the idea of using quantum computers to ``sample quantum mechanics'' has stood as a pivotal pathway to quantum advantage demonstrations. Yet a severe drawback remains: Its verification requires exponential classical computation. As an attempt to overcome this difficulty, we propose a new candidate for quantum advantage experiments with ``peaked circuits'', i.e., quantum circuits with high concentrations on a computational basis. Naturally, this heaviest output string can be used for classical verification. 
    
    In this work, we analytically and numerically study an explicit construction of peaked circuits by attaching $\tau_r$ layers of $1d$ random quantum circuits (RQCs) with $\tau_{p}$ layers of ``peaking circuits''. We study the relationship between $\tau_r$, $\tau_{p}$, and the resultant peak weight, $\delta$.
    
    First, making use of RQCs' unitary design property, we prove that getting a constant $\delta$ from such circuits requires $\tau_{p} = \Omega(\tau_r^{1/5})$ with overwhelming probability. Next, we treat the peaking layers as a parameterized quantum circuit (PQC) and numerically maximize $\delta$ at various system sizes and circuit depths. To systematically study the asymptotic behavior, we set the depth of the peaking circuits to be a constant fraction of the random circuits. If the trend in our result persists, at system size $n=50$ and for $\tau_r = n=2\tau_p$ we would be able to generate an average peak weight of $0.05\%$. This means the peaked circuits we found could serve as a potential candidate for future verifiable quantum advantage experiments.
