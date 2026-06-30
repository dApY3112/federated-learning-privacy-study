# Federated Learning Reading Map: Privacy, Optimization, and Secure Aggregation

**Purpose.** This reading map summarizes my current understanding of Federated Learning (FL), with a particular focus on privacy, optimization under heterogeneous clients, and the connection between FL and cryptography-based privacy-preserving machine learning (PPML). It is intended as a research-oriented note rather than a tutorial: the goal is to identify the main concepts, assumptions, threat models, and open questions that I want to explore further.

**Background motivation.** My previous work approaches privacy-preserving ML from a cryptographic perspective, where the goal is often to protect data, model inputs, intermediate computation, or inference through formal security protocols. Federated learning offers a complementary approach: instead of collecting training data centrally, training is coordinated across distributed clients while data remains local. This makes FL naturally relevant to private and decentralized ML, but it is important to emphasize that FL by itself is not a complete privacy guarantee. Model updates, gradients, aggregate statistics, or the final model may still leak information.

This reading map is organized around five questions:

1. What is the core FL setting?
2. Why is FedAvg the canonical starting point?
3. What makes FL technically different from ordinary distributed training?
4. Why is privacy in FL non-trivial?
5. How can cryptographic PPML, secure aggregation, and differential privacy be combined or compared?

---

## 1. Core Idea of Federated Learning

Federated learning is a machine learning setting where multiple clients collaboratively train a shared model under the coordination of a server, while each client's raw data remains local. A typical training round has the following structure:

1. The server sends the current global model to a subset of clients.
2. Each selected client trains the model locally on its own data.
3. Clients send model updates, gradients, or parameter differences back to the server.
4. The server aggregates these updates to produce the next global model.

This is different from centralized training, where all data is collected in one place before training. It is also different from standard distributed training in a data center, where workers are usually reliable, homogeneous, and trained on data partitions that are often assumed to be relatively balanced or IID.

FL is motivated by settings where data is decentralized, privacy-sensitive, large, or difficult to move. Examples include mobile devices, hospitals, banks, organizations, and edge devices. In these settings, the clients may have different amounts of data, different data distributions, different compute resources, and different availability patterns.

A key distinction is between two major FL regimes:

- **Cross-device FL:** Many clients, often mobile or edge devices. Clients may be unreliable, intermittently available, and resource-constrained. Only a small fraction of clients participate in each round.
- **Cross-silo FL:** A smaller number of organizations participate, such as hospitals or companies. Clients are usually more reliable and have more compute, but legal, privacy, or institutional constraints may prevent raw data sharing.

For my interests, cross-device FL is especially interesting because it creates a challenging intersection of privacy, scalability, partial participation, and heterogeneous data.

---

## 2. Canonical Algorithm: Federated Averaging (FedAvg)

The standard starting point for FL is **Federated Averaging (FedAvg)**, introduced by McMahan et al. in *Communication-Efficient Learning of Deep Networks from Decentralized Data*.

The high-level idea is simple:

1. The server initializes a model.
2. In each communication round, it selects a subset of clients.
3. Each client performs several local SGD steps on its own data.
4. The server averages the resulting client updates or model parameters, often weighted by the number of local examples.

The main insight of FedAvg is that local computation can reduce communication. Instead of communicating after every gradient step, clients perform multiple local updates before sending information back to the server. This is important because communication, rather than computation, is often a major bottleneck in federated systems.

However, FedAvg also introduces a key tension. More local steps may reduce communication, but they can also cause local models to drift away from one another when client data is non-IID. This makes the choice of local epochs, learning rate, client sampling rate, and aggregation rule important.

### What I learned from FedAvg

FedAvg is not merely distributed SGD. The fact that each client trains on its own local distribution changes the optimization behavior. If clients have very different data distributions, local training may move each client model toward different local optima. When the server averages these updates, the resulting global model may converge more slowly or unstably.

This makes FL a setting where optimization, statistics, systems constraints, and privacy interact. A method that works well under IID partitions may behave very differently under realistic non-IID client data.

### Questions after reading FedAvg

- How should the number of local steps be chosen when client data is highly heterogeneous?
- How does client sampling affect convergence and fairness?
- Can privacy mechanisms such as secure aggregation or DP change the optimization dynamics?
- What does it mean for a global model to perform well if clients have very different local distributions?

---

## 3. Key Challenges in Federated Learning

### 3.1 Statistical heterogeneity: non-IID data

In FL, client data is usually not IID. For example, in a keyboard prediction task, each user's typing data reflects their language, habits, topics, and social context. In a healthcare setting, hospitals may serve different populations and have different measurement practices.

Non-IID data creates several difficulties:

- Client updates may point in different directions.
- Some clients may dominate the global objective if they have more data.
- The global model may underperform for minority or unusual clients.
- Local training can create client drift.

This is one of the main reasons FL is not just a systems problem. It is also a statistical and optimization problem.

### 3.2 Client drift

Client drift occurs when local client updates move away from the direction that would be taken by centralized training. The more local epochs each client performs, the more severe this drift may become under heterogeneous data.

Algorithms such as FedProx and SCAFFOLD can be understood as attempts to reduce or correct the negative effects of heterogeneity. FedProx modifies the local objective using a proximal term, while SCAFFOLD uses control variates to correct for client drift.

### 3.3 Partial participation

In cross-device FL, only a subset of clients participates in each round. This is necessary for scalability, but it introduces uncertainty. The participating clients in one round may not represent the full population. Some clients may be unavailable due to connectivity, battery, or device constraints.

This raises both optimization and privacy questions. For optimization, partial participation can make training noisier. For privacy, participation patterns themselves may reveal information if not handled carefully.

### 3.4 Systems heterogeneity

Clients may differ in compute power, memory, network bandwidth, battery level, and availability. This creates stragglers and dropouts. A realistic FL protocol must tolerate client failure and should not assume that every selected client will complete every round.

This is one reason secure aggregation in FL is technically difficult: a secure aggregation protocol must protect individual client updates while still allowing aggregation to succeed even if some clients drop out.

### 3.5 Communication efficiency

Communication is often one of the main constraints in FL. Reducing communication can involve:

- increasing local computation,
- compressing updates,
- quantizing gradients,
- reducing the number of rounds,
- sampling fewer clients per round,
- sending sparse updates.

However, these techniques may interact with privacy. For example, compression or sparsification may change what information is exposed through updates. Adding DP noise may increase the number of rounds or reduce utility. Secure aggregation may increase communication overhead.

---

## 4. Privacy in FL Is Not Automatic

A common misunderstanding is that FL automatically guarantees privacy because raw data never leaves the client. This is not correct. FL reduces the need to centralize raw data, but model updates can still reveal sensitive information.

Potential privacy risks include:

- **Gradient leakage:** shared gradients or updates may contain information about training examples.
- **Membership inference:** an adversary may infer whether a particular example or user participated in training.
- **Property inference:** updates may reveal aggregate properties of a client's local data.
- **Reconstruction attacks:** in some settings, gradients can be used to reconstruct training inputs.
- **Final model leakage:** even if updates are protected, the trained model may memorize or reveal information about training data.

The privacy threat model matters. Different mechanisms protect against different adversaries.

### 4.1 Secure aggregation

Secure aggregation is a cryptographic protocol that allows the server to learn only the aggregate of client updates, not individual client updates. This is closely connected to my background in cryptographic PPML.

In the FL setting, secure aggregation is useful because individual updates can be highly sensitive. If the server only observes the sum or average of many updates, direct inspection of a single client's update becomes harder.

However, secure aggregation has limitations:

- It does not necessarily prevent leakage from the aggregate update.
- It does not by itself protect against leakage from the final trained model.
- It may require enough participating clients to provide meaningful aggregation.
- It must handle client dropouts and communication constraints.
- It is a protocol-level guarantee, not a statistical privacy guarantee like DP.

### 4.2 Differential privacy

Differential privacy provides a formal statistical guarantee that limits how much the output of an algorithm can change when one individual or one user's data is added or removed. In ML, DP is often implemented by clipping gradients or updates and adding calibrated noise.

In FL, DP can be applied at different levels:

- **Example-level DP:** protects individual examples within a client's dataset.
- **User-level DP:** protects the contribution of an entire client/user.

User-level DP is especially natural for cross-device FL, where each client may correspond to a person. However, user-level DP can be challenging because each user's contribution may be high-dimensional and heterogeneous.

DP and secure aggregation are complementary. Secure aggregation can hide individual updates from the server, while DP can limit what can be inferred from the aggregate output or final model. Combining them is attractive, but it raises trade-offs in utility, communication, accounting, and deployment.

### 4.3 Gradient leakage and reconstruction attacks

The Deep Leakage from Gradients line of work is important because it challenges the assumption that gradients are safe to share. If gradients can encode information about training examples, then FL protocols that expose individual updates to the server may be vulnerable.

This motivates the use of secure aggregation, DP, update clipping, compression, and careful threat modeling. It also suggests that privacy analysis in FL should not stop at the statement that raw data remains local.

---

## 5. Connection to Cryptography-Based PPML

My prior work in cryptography-based PPML gives me a useful perspective on FL. In cryptographic PPML, the goal is often to protect data or computation using formal protocols such as secure multi-party computation, homomorphic encryption, zero-knowledge proofs, or trusted execution environments. The central question is usually: what does an adversary learn from participating in or observing the protocol?

FL asks a related but different question: can we train a useful model from decentralized data without moving raw data to a central server?

The two perspectives are complementary:

- Cryptographic PPML can provide strong protocol-level confidentiality.
- FL provides a scalable training paradigm for decentralized data.
- Secure aggregation imports cryptographic ideas into FL.
- DP provides statistical privacy guarantees that can protect against leakage from outputs.
- Robust aggregation addresses malicious or unreliable client behavior.

A key lesson is that privacy-preserving ML is not a single technique. Different tools protect different parts of the learning pipeline.

| Technique | Main protection | Typical limitation |
|---|---|---|
| Federated Learning | Keeps raw data local | Updates and final model may leak information |
| Secure Aggregation | Hides individual client updates from the server | Aggregate or final model may still leak |
| Differential Privacy | Bounds influence of an individual/user on output | Can reduce utility and complicate optimization |
| Cryptographic PPML | Protects computation/data under formal protocol models | Can be computationally or communication expensive |
| Robust Aggregation | Mitigates malicious or noisy clients | May not provide privacy by itself |

This motivates my interest in hybrid approaches. For example, one can ask how secure aggregation and DP should be combined under non-IID client data, or whether cryptographic protections can be made scalable enough for realistic FL workloads.

---

## 6. Paper Reading Map

### 6.1 McMahan et al. — Communication-Efficient Learning of Deep Networks from Decentralized Data

**Main problem.** How can deep networks be trained when data is distributed across many clients and communication is expensive?

**Core idea.** FedAvg lets clients perform multiple local SGD steps before sending model updates to the server, which then averages the updates.

**What I learned.** FedAvg establishes the basic FL training loop and highlights communication efficiency as a central constraint. It also makes clear that realistic FL data is often unbalanced and non-IID.

**Why it matters for my interests.** FedAvg is the baseline I need to understand before studying privacy-preserving FL. Any privacy mechanism added to FL must be evaluated against its effect on convergence, communication, and utility.

**Questions.** How do privacy mechanisms interact with the local-update structure of FedAvg? Does adding secure aggregation or DP change the optimal choice of client sampling rate or local epochs?

---

### 6.2 Bonawitz et al. — Practical Secure Aggregation for Federated Learning on User-Held Data

**Main problem.** How can a server aggregate high-dimensional client updates without seeing any individual client's update, while tolerating client dropout?

**Core idea.** Use a secure aggregation protocol that reveals only the aggregate value to the server. The protocol is designed for high-dimensional vectors and realistic FL conditions where some clients may fail to complete the protocol.

**What I learned.** Secure aggregation is one of the most natural bridges between cryptography and FL. It addresses a concrete privacy risk: direct exposure of individual client updates to the server.

**Why it matters for my interests.** This paper connects directly to cryptographic PPML. It shows how cryptographic protocols can be adapted to the practical constraints of federated training.

**Questions.** If the server only sees aggregated updates, what privacy risks remain? How large must the aggregation group be? How should secure aggregation be combined with DP when client data is non-IID?

---

### 6.3 Zhu, Liu, and Han — Deep Leakage from Gradients

**Main problem.** Are gradients safe to share in distributed or collaborative learning?

**Core idea.** The paper shows that private training data can sometimes be reconstructed from shared gradients, challenging the intuition that gradients are harmless.

**What I learned.** FL privacy cannot be justified only by saying that raw data stays local. Updates may encode sensitive information, and privacy depends on what is shared, with whom, and under what threat model.

**Why it matters for my interests.** This motivates the need for secure aggregation, DP, and other defenses. It also gives a concrete reason why cryptographic protection of updates can matter.

**Questions.** How realistic are gradient leakage attacks under secure aggregation, client sampling, local epochs, batching, compression, or DP noise? What forms of leakage remain visible through aggregate updates or final models?

---

### 6.4 Abadi et al. — Deep Learning with Differential Privacy

**Main problem.** How can deep neural networks be trained with differential privacy?

**Core idea.** DP-SGD clips per-example gradients and adds noise, while tracking cumulative privacy loss through a privacy accountant.

**What I learned.** Differential privacy provides a formal privacy notion that is different from cryptographic confidentiality. It is not mainly about hiding messages during a protocol; it is about bounding the influence of any individual data point or user on the algorithm's output.

**Why it matters for my interests.** DP is a key complement to FL and secure aggregation. Even if cryptography protects updates during training, the final model may still leak information. DP can provide protection against this kind of output-level leakage.

**Questions.** What is the best level of privacy in FL: example-level or user-level? How much utility is lost when DP is applied under heterogeneous client distributions? Can secure aggregation make distributed DP more practical?

---

### 6.5 Li et al. — Federated Optimization in Heterogeneous Networks / FedProx

**Main problem.** How can FL handle statistical and systems heterogeneity across clients?

**Core idea.** FedProx modifies the local client objective by adding a proximal term that discourages local models from moving too far away from the global model.

**What I learned.** Heterogeneity is not a minor implementation detail. It changes optimization behavior and can make FedAvg unstable. A small modification to the local objective can improve robustness in heterogeneous settings.

**Why it matters for my interests.** Privacy mechanisms are deployed on top of an already difficult optimization problem. If non-IID data makes training unstable, adding clipping, noise, or secure protocols may further complicate training.

**Questions.** How should privacy-preserving FL algorithms account for systems and statistical heterogeneity? Does DP noise interact differently with FedAvg and FedProx?

---

### 6.6 Karimireddy et al. — SCAFFOLD

**Main problem.** Why does FedAvg suffer under heterogeneous data, and how can client drift be corrected?

**Core idea.** SCAFFOLD uses control variates to correct local updates and reduce client drift caused by non-IID data.

**What I learned.** Client drift is a central phenomenon in FL optimization. Correcting drift can improve convergence and reduce communication rounds.

**Why it matters for my interests.** If privacy mechanisms introduce additional noise or restrict what updates can be shared, optimization methods that are robust to client drift may become even more important.

**Questions.** Can drift correction be combined with secure aggregation? Do control variates introduce additional privacy leakage? How would DP accounting work if auxiliary correction terms are maintained across rounds?

---

### 6.7 Kairouz et al. — Advances and Open Problems in Federated Learning

**Main problem.** What are the major research directions, challenges, and open problems in FL?

**Core idea.** FL is a broad research area involving optimization, privacy, security, fairness, personalization, systems, and applications.

**What I learned.** FL should not be viewed as a single algorithm. It is a research framework with many interacting concerns. Privacy is only one part of a larger design space.

**Why it matters for my interests.** This survey helps me place privacy-preserving FL within the broader landscape. It also helps identify where my cryptographic background can contribute: secure aggregation, threat modeling, privacy attacks, and hybrid privacy mechanisms.

**Questions.** Which open problems are best suited for someone with a background in cryptography-based PPML? How can formal security thinking improve the design and evaluation of FL systems?

---

## 7. Research Directions I Am Interested In

### 7.1 Secure aggregation plus differential privacy

Secure aggregation and DP protect different things. Secure aggregation hides individual updates from the server, while DP limits what can be inferred from the output. I am interested in how to combine these tools in a way that is practical under non-IID data and partial client participation.

Possible research question:

> How can secure aggregation and user-level DP be combined in FL without excessive utility loss under heterogeneous client data?

### 7.2 Privacy leakage despite local data storage

FL keeps data local, but local storage alone does not eliminate privacy leakage. I am interested in studying what information can still be inferred from updates, aggregate updates, or final models.

Possible research question:

> What privacy risks remain when individual client updates are hidden but aggregate updates and final models are visible?

### 7.3 Threat models for privacy-preserving FL

Cryptographic PPML often uses explicit adversarial models, such as semi-honest or malicious adversaries. FL papers sometimes use different assumptions, including honest-but-curious servers, malicious clients, or external attackers. I am interested in making these assumptions more explicit and comparing what each privacy mechanism actually protects.

Possible research question:

> How should threat models in FL distinguish between protocol-level confidentiality, statistical privacy, and model-level leakage?

### 7.4 Non-IID data and privacy-utility trade-offs

Non-IID data affects both optimization and privacy. For example, rare client distributions may be more identifiable, and DP noise may affect minority clients differently.

Possible research question:

> How does statistical heterogeneity affect privacy leakage and fairness in federated training?

### 7.5 Bridging cryptographic PPML and scalable FL

Cryptographic protocols can provide strong privacy, but they may be expensive. FL provides a scalable training paradigm, but it may not provide formal privacy by itself. I am interested in hybrid approaches that combine the strengths of both.

Possible research question:

> Can cryptographic techniques be integrated into FL training in a way that remains practical for high-dimensional models and unreliable clients?

---

## 8. Practical Mini-Project Plan

To demonstrate hands-on understanding, I would like to implement a small FL study with the following structure.

### Project title

**FedAvg under Non-IID Data: Optimization Behavior and Privacy Limitations**

### Core implementation

- Implement FedAvg in PyTorch.
- Use MNIST, Fashion-MNIST, or CIFAR-10.
- Simulate multiple clients.
- Compare centralized training, IID FedAvg, and non-IID FedAvg.
- Track accuracy, loss, and communication rounds.

### Experiments

1. **IID split:** each client receives a roughly similar label distribution.
2. **Label-skew non-IID split:** each client receives data from only a few labels.
3. **Vary local epochs:** compare 1, 5, and 10 local epochs.
4. **Vary client participation:** compare full participation with partial participation.
5. **Optional privacy extension:** add update clipping and simple Gaussian noise to approximate the utility impact of DP-style training.

### Expected observations

- FedAvg should work reasonably well under IID splits.
- Non-IID splits may slow convergence or reduce accuracy.
- More local epochs may reduce communication but worsen client drift under non-IID data.
- Adding noise may reduce utility, especially when data is heterogeneous or the number of clients per round is small.

### Privacy discussion to include in the report

- FL keeps raw data local, but updates may leak information.
- Secure aggregation can hide individual updates from the server.
- DP can limit the influence of individual users or examples on the final output.
- Crypto-based PPML, FL, secure aggregation, and DP should be viewed as complementary tools.

---

## 9. Summary of My Current Understanding

My current understanding is that federated learning is a privacy-motivated but not automatically privacy-preserving training paradigm. Its core contribution is to enable collaborative training over decentralized data, but the privacy guarantees depend on additional mechanisms and careful threat modeling.

FedAvg is the canonical baseline because it captures the central FL trade-off between local computation and communication. However, realistic FL is difficult because clients are heterogeneous, only partially available, and often statistically non-IID. These factors create optimization challenges such as client drift.

From a privacy perspective, FL should be combined with mechanisms such as secure aggregation and differential privacy. Secure aggregation is especially connected to my cryptographic background because it protects individual client updates during aggregation. DP is complementary because it can protect against leakage from aggregate outputs or final models.

The most interesting research direction for me is the intersection of cryptographic privacy, federated optimization, and statistical privacy. I want to understand how formal privacy mechanisms behave under realistic FL conditions such as non-IID data, partial participation, client dropout, and communication constraints.

---

## 10. References

1. H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Agüera y Arcas. **Communication-Efficient Learning of Deep Networks from Decentralized Data.** AISTATS 2017. arXiv:1602.05629.  
   https://arxiv.org/abs/1602.05629

2. Keith Bonawitz, Vladimir Ivanov, Ben Kreuter, Antonio Marcedone, H. Brendan McMahan, Sarvar Patel, Daniel Ramage, Aaron Segal, and Karn Seth. **Practical Secure Aggregation for Federated Learning on User-Held Data.** NIPS Workshop 2016 / CCS 2017 version. arXiv:1611.04482.  
   https://arxiv.org/abs/1611.04482

3. Ligeng Zhu, Zhijian Liu, and Song Han. **Deep Leakage from Gradients.** NeurIPS 2019. arXiv:1906.08935.  
   https://arxiv.org/abs/1906.08935

4. Martín Abadi, Andy Chu, Ian Goodfellow, H. Brendan McMahan, Ilya Mironov, Kunal Talwar, and Li Zhang. **Deep Learning with Differential Privacy.** CCS 2016. arXiv:1607.00133.  
   https://arxiv.org/abs/1607.00133

5. Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, and Virginia Smith. **Federated Optimization in Heterogeneous Networks.** MLSys 2020. arXiv:1812.06127.  
   https://arxiv.org/abs/1812.06127

6. Sai Praneeth Karimireddy, Satyen Kale, Mehryar Mohri, Sashank J. Reddi, Sebastian U. Stich, and Ananda Theertha Suresh. **SCAFFOLD: Stochastic Controlled Averaging for Federated Learning.** ICML 2020. arXiv:1910.06378.  
   https://arxiv.org/abs/1910.06378

7. Peter Kairouz, H. Brendan McMahan, Brendan Avent, Aurélien Bellet, Mehdi Bennis, and many others. **Advances and Open Problems in Federated Learning.** Foundations and Trends in Machine Learning, 2021. arXiv:1912.04977.  
   https://arxiv.org/abs/1912.04977

8. H. Brendan McMahan, Daniel Ramage, Kunal Talwar, and Li Zhang. **Learning Differentially Private Recurrent Language Models.** ICLR 2018. arXiv:1710.06963.  
   https://arxiv.org/abs/1710.06963
