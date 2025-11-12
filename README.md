# **Spectral Bipartition for Community Detection in the Karate Club Network**

This project implements the **Spectral Bipartition method** for community detection on the classic **Zachary’s Karate Club network**.  


Through this method, we learn to identify **meaningful communities** within a network and gain the ability to **predict how communities form and evolve**.  
By analyzing **metric evolution**, we can also determine the **most influential nodes** that shape the structure of these communities.

Additionally, we apply an **optimization technique based on the Rayleigh–Ritz theorem**, which significantly reduces computational complexity and improves algorithmic efficiency.

Finally, by implementing the **Recursive Spectral Bipartitioning** approach, we iteratively uncover **sub-communities within larger groups** until no further meaningful divisions can be detected — revealing the full hierarchical structure of the network.

---

### **Included in this Notebook**
- Construction of the **Adjacency** and **Probability** matrices  
- Calculation of **Modularity scores**  
- Analysis of **Centrality measures** (Degree, Betweenness, Closeness, and Clustering)  
- **Visualization** of community partitions and metric evolution across recursive splits  

---

### **Reference**

The modularity objective, modularity matrix formulation, and spectral bipartition algorithm are based on standard results from:

> **Newman, M. E. J. (2006).** *Modularity and community structure in networks.*  
> *Proceedings of the National Academy of Sciences (PNAS)*, 103(23), 8577–8582.  
> [https://doi.org/10.1073/pnas.0601602103](https://doi.org/10.1073/pnas.0601602103)

