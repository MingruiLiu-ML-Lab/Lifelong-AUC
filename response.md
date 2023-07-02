# Dear Reviewer JJqN,thank you for your valuable feedback

1. The gradient interference issue and its impact on training and overall performance have not been sufficiently explored and feel very speculative. While the experiments confirm the advantage of using TWO-MODEL over ONE-MODEL(with and without model variable splitting), it is not an indication of the gradient interference issue.
> Regarding the issue of gradient interference, we agree that this is an area that requires further exploration and investigation. Our study was intended to shed light on the potential impact of this issue on training and performance, and we acknowledge that further research is needed to more fully understand its implications. 
> 
> In Appendix Fig 5 and Fig 6, we further explore gradient interference by analyzing gradient angles before and after the datasets are made imbalanced. Hope this would give a sight of gradient interference problem.

2. In the setting of real-world imbalanced data for lifelong learning, the distribution of classes(especially the imbalanced ratio) may vary over tasks. It is not clear how this should affect the functions \lambda_1(.) and \lambda_2(.) that weigh the AUC from the current task and replay buffer.
   
> These parameters are adaptively changing according to the AUC value on the reference data and the current data. The distribution over classes over different tasks will affect AUC, so it will also affect parameters $\lambda_1$ and $\lambda_2$ dynamically. Similar ideas were also used in classical continual learning papers (MEGA, A-GEM, GEM): the only difference is that classical setting adaptively changes these paraemters by cross-entropy loss instead of AUC.

> Following MEGA [ref1], we choose λ1 = AUCref(w)/AUCt(w) and λ2 = 1 based on the model performance. In our experiments, the weigh λ1 varys from 0.1 to 10 at most of the time. We will include this observation in the revised version.

> [ref1] Yunhui Guo, Mingrui Liu, Tianbao Yang, and Tajana Rosing.Improved schemes for episodic memory-based lifelong learning. Advances in Neural Information Processing Systems, 33, 2020


3. The use of average AUC as an evaluation metric has not been justified, and it is not clear why this should be the primary metric of interest. It is particularly necessary as a) the definition of "positive" and "negative" class changes with each task in the experiments, and b) AUC metric is not inherently additive.
> We consider both binary (Table 1，2) and multi-class AUC (Table 3).

>In binary setting, we categorize positive and negative classes before the continual learning process, so they are inherently additive. For example, in Split-CIFAR, the negative class is defined as classes $\{0+i*n\sim 3+i*n\}_{i=1}^{T}$ in the original CIFAR100 dataset, where $T$ is the total number of tasks, $n$ is the number of classes in each task in the original Split CIFAR-10 dataset. The rest of classes are all defined as positive.

>In multi-class setting, we use multiclass AUC [ref1] as a metric instead of binary AUC, so they are inherently additive as well. We will make it more clear in the revised version.

> [ref1] David J Hand and Robert J Till. A simple generalisation of the area under the roc curve for multiple class classification problems. Machine learning, 45(2):171–186, 2001.

4. My key reservations are in regard to the gradient interference issue as pointed out above. In the current experiments comparing the gradient angles for balanced and imbalanced case, my interpretation is that the distribution has a larger variance i.e. there are more occurrences of smaller as well as larger angles in the imbalanced scenario compared to the balanced case.

> The imbalanced case is characterized by more obtuse angles and larger variance (standard deviation) as well. To investigate this further, we analyzed the gradient angles and present the results in Appendix Figures 5 and 6. Figure 5 illustrates that the imbalanced case has a larger variance. Figure 6 shows that after making the dataset imbalanced, the proportion of obtuse angles increased from 1.05% to 3.16% on Split-CUB200 and from 0.13% to 9.68% on Split-CIFAR100.

> In our interpretation, both larger variance and obtuse angles impede the algorithm to achieve good performance. We used obtuse angles as an indication of gradient interference since gradients are in conflict when the angle is larger than 90 degrees. We believe larger variance would also contirbute to gradient interference problem due to its dramatic changes on gradients direction.

  
5. Moreover, the exact experimental setting for these experiments has not been specified. I'm particularly concerned in terms of how the comparison can be made fair in terms of choosing the sample size between balanced and imbalanced cases.
> The sampling size is same as the batch size used on current task, namely 64 on Split-CIFAR100, Split-CUB200, and Split-AWA2. To clarify, when running on Split-CIFAR100, the replay buffer feeds 64 samples and the current data stream feeds 64 samples.
> We acknowledge that we did not explicitly state the batch size of sampling in our paper, and we apologize for any confusion caused. We will update the paper to provide more explicit details on the experimental settings

6. Another key aspect that I feel has not been clearly addressed is the choice of AUC metric, both for training and evaluation. Concerns about the evaluation part are mentioned above. For the training part, I am curious if the use of class-balanced loss(reweighting) has been used in the literature since that is the classical method to deal with the imbalance. It could also be interesting to compare empirically against the current methods.
   
> We are wondering if the reviewer can provide the literatures of class-balanced loss. We indeed compare with a similar setting, which is the class-balanced sampling ablation in Appendix Table 1. We showed that our algorithm is better than the class-balanced sampling baseline. 

7. The optimization problem in this work is peculiar and perhaps some convergence plots or explanation regarding the optimization procedure in the Appendix could be of interest to the readers.

> Please note that the optimization problem is indeed maximizing AUC. So the plots in Figure 3 is actually the convergence plots of the loss function.

8. I believe there needs to be a description of lifelong learning in general and the related challenges very early in the introduction so that the readers not familiar with them can get some context.
> Thank you for suggestions, in the next version of our manuscript, we will add a brief section in the introduction that defines lifelong learning and highlights the challenges associated with it. 

9. Some recent works related to imbalanced lifelong learning have not been discussed/cited - check the references below. 
> We appreciate your suggestion to include references to recent works related to imbalanced lifelong learning that were not discussed in our manuscript. We have reviewed the papers you provided and agree that they are relevant to our work. We will cite and disccuss them in revised version.

10. There are some formatting errors in references Sec 2.2, the name of authors seems to be listed twice.
> Thanks for mentioning it. We will fix it in the revision.

# Dear Reviewer FZuP, Thank you for your feedback
1. The method introduced by the authors to solve the gradient interference problem is similar to the method in Gradient Surgery for Multi-Task Learning [A]. The work should be added for this, and additional explanation should be included regarding its relevance.
> Thank you for your suggestion. We will mention and cite this relevant work in the next version. TODO: GUOYUNHUI

2. An experiment may be necessary to investigate the correlation between the lambdas and beta in formula (9). To facilitate other researchers in extending this study, an explanation of how to set beta appropriately should be included.
> \lambda is adaptively changed according our algorithm so practioners do not need to tune $\lambda$. Now we provide some results in varying  $\beta$ because $\beta$ is a tuned hyperparameter which controls the tradeoff of maximizing AUC and alignment of two models. We find that the best tradeoff is choosing $\beta$ to be $0.1$ in various benchmarks, so we suggest practioners to use $\beta=0.1$ for their own tasks.
> We will include the ablation in the next version.

>| $\beta$ |CIFAR100|CUB200|AWA2|
>|---------|-------|------|-----|
>| 0.1     |68.4|70.4|98.2|
>| 0       |66.09|72.03|95.8|
>| 0.01    |65.8|69.6|93.7|
>| 1.0     |63.6|51.11|54.1|
3. To improve readability, please change the distorted plots in Figure 3 and adjust the spacing between the caption and algorithm in Figure 2.
> We appreciate your suggestions for improvement. We will work on improving the quality in the final version of the paper to make them more clear and readable.
4. For completeness of the paper, an ablation study on hyperparameters needs to be included.
> Thank you for your suggetions, we will incorporate an ablation study on hyperparameters in the revised version of the paper. Ablation on learning rate and batch size are shown below.
>
> | lr   | CIFAR100| CUB200|
> |------|--------| -----|
> | 0.1  | 67.08  | 70.4
> | 0.03 | 66.52  | 56.2
> | 0.3  | 51.88  | 81.1
> 
> | batch | CIFAR100 | CUB200|
> |----------|--------|----|
> | 32 | 61.08| - |
> | 64 | 67.08| 70.4
> | 128| 65.16| 65.0

# Dear Reviewer t6Vd, Thank you for your feedback
1. The proposed method has a hyperparameter \beta (the alignment penalty). It is unclear how this parameter is set in the experiments and how its value should be determined in practice. Also, the proposed approach is not evaluated with \beta = 0, which should be part of the ablation study.
   
> We provide some results in varying  $\beta$, which is a tuned hyperparameter which controls the tradeoff of maximizing AUC and alignment of two models. We find that the best tradeoff is choosing $\beta$ to be $0.1$ in various benchmarks, so we suggest practioners to use $\beta=0.1$ for their own tasks. $\beta=0$ does not lead to good performance because it does not promote the alignment of two models.
>
>|beta|CIFAR100|CUB200|AWA2|
>|------|-------|------|-----|
|0.1|68.4|70.4|98.2|
|0|66.09|72.03|95.8|
|0.01|65.8|69.6|93.7|
|1.0|63.6|51.11|54.1|

2. Ablation study  of the two \lambda parameters. It is unclear how much of the improvement in performance is due to these adaptively chosen weights.
> TODO: HAOJIE, debug alpha clip
> We investigated the effect of the different values of $\lambda_2/\lambda_1$. 
>
>| lambda   | CIFAR100 | CUB200   | AWA2     |
>|----------|----------|----------|----------|
>| Adaptive | 72.6±1.6 | 70.4±1.2 | 98.9±0.4 |
>| 1:1      | 72.5±0.9 | 71.1±1.3 | 98.7±0.7 |
>| 1:0.5    | 73.5±0.9 | 72.3±1.3 | 98.9±0.6 |
>| 0.5:1    | 72.5±1.6 | 69.8±1.3 | 98.5±0.8 |



3. The submission should also compare to implementing AUROC maximization (instead of cross-entropy minimization) in one or two of the chosen competitors (e.g., DER and EWC ?).
> Existing literatures like ewc and der benefits from AUC maximization, which further validate the advantage of AUC maximiazation in imbalanced setting.
> However, our proposed method achieve superior performance compared to these existing approaches, owing to the incorporation of an additional model decoupling mechanism.
>
>|EWC-auc |CIFR100|CUB200|AWA2|
>|------|-------|------|-----|
>|EWC|64.4|51.5|56.2|
>| EWC-auc |62.4|69.7|97.1|
>| DER| 62.6|65.7|80.4|
>| DER-auc |70.1|62.1|91.5|
>| DIANA | 68.4|70.4|98.2|

4. The submission does not state explicitly how predictions are made with the proposed model.
> The prediction is made by calculating the prediction score (which is calculated by forward porpoagation of neural networks), if the score is higher than the threshold then we predict it to be positive, otherwise it is negative. For multi-class AUC case, we use the standard one-versus-all approach to do the classification.

# Dear reviewer USzp, Thank you for your feedback
1. As far as I can tell, the main optimization framework (Eq.5) is highly based on (Guo et al. 2020) as pointed out in Section 4.1. The novel contribution from this paper seems to be Eq.8 and its relaxation version Eq.9, which I think is slightly straightforward to derive
> The standard online AUC maximization only uses one model and it is insufficient for continual learning. We are the first to demonstrate the gradient interference issue of the single model and address this issue by introducing two models and model alignment.
>
> In addition, the $\beta$ does not need to be tuned extensively in pracitice. For example, we show that $\beta=0.1$ is the default good parameter for all of our experiments. Please check our experiments below, where the numbers indicate the value of AUC.

>|beta|CIFAR100|CUB200|AWA2|
>|------|-------|------|-----|
>|0.1|68.4|70.4|98.2|
>|0|66.09|72.03|95.8|
>|0.01|65.8|69.6|93.7|
>|1.0|63.6|51.11|54.1|

2. training double models is definitely more costly, let alone the need to carefully tune the alignment hyper-parameter.
> Yes it's correct that model decoupling with an alignment penalty can be computationally expensive compared with training one model. However, this approach can often yield better results and faster than using a single model with a complex function. According to Talbe 4 Time and space complexity, although our approach cost 34.5s, it still faster than EWC, MAS and GEM, which cost 78.6s, 77.3s and 109s respectively.

3. Why choose AUC maximazation should be justified.
> Previous lifelong learning literature use accuracy as default metrics. However, when the data stream becomes very imbalanced, AUC becomes more informative than Accuracy. For example, if the imbalanced data ratio is $99:1$, then a naive classifier which classify every example to be positive has $99\%$ accuracy but it is definitely not a good classifier. In this case, AUC is much more informative and we should directly optimize AUC rather than accuracy.

4.  In addition, results on multi-class imbalanced classification is more worthwhile to report rather than those in the balanced case.
> Thank you for your suggestions, we will include the ablation on multi-class imbalanced classification in the next version
> 
>| method |CIFAR100| CUB200 | AWA2|
> |------|--------|------| ----|
>| single    |64.4| 57.1   |52.8
>| ewc     |62.7| 54.7   |60.7
>| mas      |63.3| 54.6   |56.6
>| gdumb   |59.6| 69.7   |84.1
>| auc      |68.2| 59.1   |82.4|


