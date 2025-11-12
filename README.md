# Modularity
# 🧩 **Spectral Bipartition for Community Detection in the Karate Club Network**

This project implements the **Spectral Bipartition method** for community detection on the classic **Zachary’s Karate Club network**.  

The approach leverages the **modularity maximization framework** introduced by *Newman (2006)* to identify meaningful community structures within a network.  
By performing **eigenvalue decomposition of the modularity matrix ($B$)**, the algorithm finds an optimal division of nodes into two communities — where nodes within a community are more densely connected than would be expected by chance.  

After the initial bipartition, the method is applied **recursively** to each detected community to explore deeper hierarchical structures and evaluate potential sub-communities. This recursive spectral approach provides a detailed view of how modular organization emerges in social networks.

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

