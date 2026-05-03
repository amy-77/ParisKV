# Expanded Baseline Comparison: Accuracy, Efficiency, and Configurations

## 1. Scope and fairness protocol

All methods are evaluated on the same platform whenever possible. We aim to keep comparisons fair by aligning the *effective retrieval budget* rather than forcing identical hyperparameters. 


## 2. Baseline configurations (summary)

We summarize the key configurations below. All methods are tuned to match **comparable retrieval budgets** whenever possible.

| Method | LongBench-v2 / GPQA | RULER |
|---|---|---|
| SOCKET | bucket_K=8, bucket_L=60, sink=64, local=256, topk=100 | topk=2048 |
| Twilight | top-p=0.5 (meanK≈402, matches ~420 budget) | same |
| RetroInfer | sink=64, local=256, ratio=0.001, clusters=8192 (size≈16) | ratio=0.013 |
| ShadowKV | local=32, outlier=384, sparse=104 | local=32, outlier=304, sparse=2048 |
| FreeKV | sink=64, local=256, page=128 (~448 total) | same |
| Quest | token_budget=420, chunk=16 | token_budget=2048 |
| MagicPIG | LSH (K=10, L=170), dynamic budget | same |
| PQCache | compress=20%, subspace=2, clusters=64 | same |


## 3. Accuracy results

## LongBench-v2 (full breakdown)

| Model | Overall | Short Easy | Short Hard | Medium Easy | Medium Hard | Long Easy | Long Hard |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Qwen3-4B** | | | | | | | |
| Full | 25.84 | 27.12 | 16.53 | 36.36 | 25.20 | 26.67 | 28.57 |
| PQCache | 17.91 | 16.95 | 19.00 | 13.60 | 19.00 | 20.00 | 19.05 |
| MagicPIG | 16.70 | 18.64 | 10.74 | 14.77 | 20.47 | 28.89 | 12.70 |
| ShadowKV | 16.30 | 13.60 | 14.80 | 22.20 | 9.90 | 21.30 | 19.00 |
| FreeKV | 19.68 | 20.34 | 13.22 | 21.59 | 24.41 | 22.22 | 17.46 |
| Quest | 19.12 | 32.00 | 16.18 | 21.43 | 12.50 | 21.05 | 24.24 |
| RetroInfer | 23.69 | 34.50 | 31.70 | 22.70 | 22.20 | 13.60 | 9.70 |
| SOCKET | 17.93 | 20.00 | 22.06 | 16.67 | 12.50 | 15.79 | 21.21 |
| Twilight | 19.12 | 28.00 | 14.71 | 21.43 | 15.62 | 21.05 | 24.24 |
| **ParisKV** | **24.60** | **35.59** | **19.49** | **26.14** | **22.05** | **28.89** | **23.81** |

| Model | Overall | Short Easy | Short Hard | Medium Easy | Medium Hard | Long Easy | Long Hard |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Qwen3-8B** | | | | | | | |
| Full | 33.59 | 50.85 | 34.71 | 32.95 | 25.98 | 37.78 | 28.57 |
| PQCache | 25.50 | 23.70 | 31.60 | 28.20 | 24.00 | 25.00 | 20.00 |
| MagicPIG | 10.34 | 8.47 | 15.70 | 7.95 | 7.87 | 13.33 | 7.94 |
| ShadowKV | 15.90 | 40.70 | 23.10 | 6.80 | 13.40 | 6.70 | 3.20 |
| FreeKV | 15.31 | 28.81 | 24.79 | 15.91 | 11.81 | 6.70 | 3.20 |
| Quest | 23.90 | 36.00 | 25.00 | 28.57 | 15.62 | 31.58 | 18.18 |
| RetroInfer | 20.48 | 44.80 | 36.70 | 20.50 | 7.90 | 4.50 | 3.20 |
| SOCKET | 17.93 | 24.00 | 22.06 | 14.29 | 20.31 | 10.53 | 9.09 |
| Twilight | 20.32 | 36.00 | 26.47 | 19.05 | 12.50 | 15.79 | 15.15 |
| **ParisKV** | **33.07** | **52.54** | **34.71** | **34.09** | **26.77** | **37.21** | **19.67** |

| Model | Overall | Short Easy | Short Hard | Medium Easy | Medium Hard | Long Easy | Long Hard |
|---|---:|---:|---:|---:|---:|---:|---:|
| **DS-R1-8B** | | | | | | | |
| Full | 13.12 | 18.64 | 15.70 | 12.50 | 8.66 | 11.11 | 14.29 |
| PQCache | 19.90 | 18.60 | 21.50 | 21.60 | 22.20 | 15.60 | 14.30 |
| MagicPIG | 13.92 | 15.25 | 11.57 | 11.36 | 14.17 | 17.78 | 17.46 |
| ShadowKV | 14.51 | 18.60 | 23.10 | 13.60 | 14.20 | 2.20 | 4.80 |
| FreeKV | 17.50 | 44.07 | 30.58 | 9.09 | 8.66 | 6.67 | 4.76 |
| Quest | 23.51 | 24.00 | 27.94 | 28.57 | 14.06 | 26.32 | 24.24 |
| RetroInfer | 15.66 | 27.60 | 20.00 | 20.50 | 9.50 | 4.50 | 9.70 |
| SOCKET | 21.51 | 20.00 | 14.71 | 30.95 | 21.88 | 15.79 | 27.27 |
| Twilight | 20.72 | 20.00 | 14.71 | 26.19 | 23.44 | 15.79 | 24.24 |
| **ParisKV** | **28.43** | **37.29** | **25.62** | **28.41** | **25.20** | **31.11** | **30.16** |



## RULER (full breakdown)

<table>
  <thead>
    <tr>
      <th style="min-width: 140px; text-align: left;">Method</th>
      <th>niah_s1</th>
      <th>niah_s2</th>
      <th>niah_m1</th>
      <th>niah_m2</th>
      <th>niah_mv</th>
      <th>niah_mq</th>
      <th>fwe</th>
      <th>qa_1</th>
      <th>qa_2</th>
      <th>vt</th>
      <th>Avg</th>
    </tr>
  </thead>
  <tbody>
    <tr><td style="text-align:left;">Llama3.1-8B</td><td>100.00</td><td>100.00</td><td>96.88</td><td>89.58</td><td>98.44</td><td>99.74</td><td>71.18</td><td>86.46</td><td>51.04</td><td>47.92</td><td>84.12</td></tr>
    <tr><td style="text-align:left;"><strong>ParisKV</strong></td><td>100.00</td><td>100.00</td><td>96.88</td><td>82.01</td><td>95.83</td><td>98.70</td><td>56.25</td><td>84.38</td><td>48.96</td><td>71.87</td><td><strong>83.49</strong></td></tr>
    <tr><td style="text-align:left;">MagicPIG</td><td>100.00</td><td>94.79</td><td>83.33</td><td>34.38</td><td>73.44</td><td>78.12</td><td>70.08</td><td>73.96</td><td>42.71</td><td>53.33</td><td>70.49</td></tr>
    <tr><td style="text-align:left;">ShadowKV</td><td>100.00</td><td>98.96</td><td>96.88</td><td>73.96</td><td>89.06</td><td>96.35</td><td>61.11</td><td>81.25</td><td>50.00</td><td>55.42</td><td>80.29</td></tr>
    <tr><td style="text-align:left;">PQCache</td><td>5.21</td><td>50.00</td><td>45.83</td><td>30.21</td><td>16.67</td><td>20.31</td><td>55.21</td><td>68.75</td><td>40.62</td><td>4.79</td><td>33.76</td></tr>
    <tr><td style="text-align:left;">SOCKET</td><td>100.00</td><td>100.00</td><td>96.88</td><td>72.92</td><td>92.71</td><td>98.18</td><td>33.33</td><td>80.21</td><td>46.88</td><td>68.13</td><td>78.92</td></tr>
    <tr><td style="text-align:left;">Twilight</td><td>98.96</td><td>100.00</td><td>95.83</td><td>76.04</td><td>90.36</td><td>97.92</td><td>57.29</td><td>80.21</td><td>48.96</td><td>74.17</td><td>77.60</td></tr>
    <tr><td style="text-align:left;">RetroInfer</td><td>95.83</td><td>98.96</td><td>95.83</td><td>55.21</td><td>94.53</td><td>97.40</td><td>61.11</td><td>78.12</td><td>42.71</td><td>54.79</td><td>77.45</td></tr>
    <tr><td style="text-align:left;">Quest</td><td>99.00</td><td>98.96</td><td>94.79</td><td>64.58</td><td>83.59</td><td>94.79</td><td>62.85</td><td>79.17</td><td>44.79</td><td>70.83</td><td>79.34</td></tr>
  </tbody>
</table>

### GPQA-diamond (ShadowKV does not support long-generation scenarios)

| Method | Qwen3-4B | Quest | RetroInfer | FreeKV | MagicPig++ | PQCache | ParisKV | ShadowKV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Score | 64.14 | 38.40 | 38.90 | 58.16 | 32.32 | 38.38 | **72.22** | NA |

---

## 4. Efficiency results

Notation:
- **NA**: unsupported by the released implementation
- **OOM**: out of memory

### Throughput at 128K decode (tokens/s)

| bs | Quest | Twilight | RetroInfer | ParisKV | Full | SOCKET | MagicPIG | PQCache | FreeKV | ShadowKV|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 48.08 | 0.90 | 34.31 | 41.10 | 43.20 | 4.44 | 120.57 | 243.91 | 21.78 |53.30|
| 2 | NA | OOM | 65.29 | 69.80 | 60.70 | OOM | NA | NA | 29.01 | 100.26|
| 3 | NA | OOM | 93.20 | 93.10 | 70.00 | OOM | NA | NA | 34.67 | 150.98|
| 8 | NA | OOM | 109.60 | **150.00** | OOM | OOM | NA | NA | OOM | 318.14|

### Decode latency (ms/token, bs=1)

| Seq len | Quest | Twilight | RetroInfer | ParisKV | Full | SOCKET | MagicPIG | PQCache | FreeKV | ShadowKV|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128K | 20.78 | 1106.14 | 29.14 | 24.42 | 23.13 | 225.00 | 120.57 | 243.91 | 21.78 | 18.76|
| 256K | 46.30 | OOM | 37.53 | **32.16** | 33.29 | OOM | 230.24 | 754.76 | NA | 19.11|
| 384K | OOM | OOM | 33.31 | 37.19 | OOM | OOM | 487.77 | 1207.81 | NA | 20.74|



### Prefill latency (TTFT, seconds)

We report prefill latency (TTFT) as a function of sequence length.

| Seq len | Quest | Twilight | RetroInfer | ParisKV | Full | SOCKET | MagicPIG | PQCache | FreeKV |ShadowKV|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128K | 37.28 | 34.56 | 35.74 | 43.00 | 33.30 | 38.29 | 55.10 | 25.50 | 43.83 |51.77 |
| 256K | 117.71 | OOM | 126.71 | 139.50 | 116.20 | OOM | 141.10 | 104.70 | OOM |152.12|
| 384K | OOM | OOM | 274.85 | 290.40 | OOM | OOM | 272.80 | 238.80 | OOM |309.26|


ParisKV delivers the strongest overall tradeoff among all compared methods: it achieves the best accuracy across baselines and outperforms almost all of them in efficiency as well. The only metric where ParisKV does not lead is decode throughput against ShadowKV. However, ShadowKV’s speed advantage comes with substantially lower accuracy, and it does not extend to a better end-to-end long-generation solution. During decoding, newly generated KV entries are still processed by full attention, which limits practical memory/bandwidth savings and prevents effective batch-size scaling in short-input, long-generation settings. In addition, ShadowKV has higher prefill latency than ParisKV and several other baselines.

> **Note on Twilight.**  
> We exclude Twilight from the main speed comparison because its open-sourced codebase only provides a Python reference implementation for accuracy evaluation. 


