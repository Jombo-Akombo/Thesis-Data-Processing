This repository contains 12 data files from 3 runs across 4 AI search tools - Claude, ChatGPT, Perplexity, and Gemini. The files contain nmumerical loan terms scraped from the web. The sets are labelled 1, 2, and 3. 

It also contains the ground truth as reported from loan providers own websites. 

Five primary metrics are calculated

- Hits - when the recorded data = ground truth
- Overshoots - data > truth
- Undershoots - data < truth
- Omissions - no data recorded but truth is recorded
- Hallucinations - data recorded but no truth recorded

And two secondary metrics

- Hit percentage
- Mean Absolute Percentage Error

They are published to files names Metrics 1-3 and Mape 1-3

There are also several - more esoteric - measures calculated

- Spearman's rank correlation
- Fisher's exact test 
- Variance ratio
- Consumer cost case
- Citation analysis
