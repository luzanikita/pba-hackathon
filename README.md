# Overview

It should be a service that provides access to some off-chain ML computations via smart contracts.
The users should have the ability to request ML model execution (for example, an LLM-based agent) in exchange for some reward (tokens).
In addition to ML model execution, the worker computes a ZK proof to prove the integrity of computations and the model version used (otherwise the worker can cheat and use some other model).
For ZK verification, we use [EZKL](https://docs.ezkl.xyz/getting-started/verify/#instructions-for-on-chain-verification).
After the user receives and verifies the response from the worker, it approves it and releases the locked tokens in the smart contract.
