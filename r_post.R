#install.packages("tidyr",repos="http://cloud.r-project.org")
library(ggplot2)
library(nlme)
library(emmeans)
library(ggpubr)
library(tidyr)

#data <- read.table("genimix_ii_test.csv",header=FALSE,sep=",",na.strings="NA",dec=".")
#colnames(data) <- c('id','genimix','ii','treat','time','std')
#genimix_std <- 1
#ii_std <- 1
data <- read.table("genimix_out.csv",header=FALSE,sep=",",na.strings="NA",dec=".")
colnames(data) <- c('id','genimix','genimix_std','ii','ii_std','treat','time')

data$treat <- as.factor(data$treat)
                                        #data$group <- as.factor(data$group)

# CUTS
data <- data[data$treat != 'nil',]
#data <- data[data$genimix > -10000,]
#data <- data[data$treat != 'scr_aso',]
#print (min(data$time))
#print (max(data$time))
#quit()
#data <- data[data$time < 120,]
#data <- data[data$treat!=-1,]

#print (summary(lm(genimix ~ time + treat + group, data=data)))
ggplot(data, aes(x=time, y=genimix, colour=treat, shape=treat)) +
    geom_point() +
    geom_errorbar(aes(ymin=genimix-genimix_std, ymax=genimix+genimix_std), width=.2,
                  position=position_dodge(0.05)) +
      labs(
       x="time", y="genimix") +

  geom_smooth(method='lm')

model <- lme(genimix ~ time * treat, random = ~1 | id, na.action="na.omit", data=data)
anova(model)
pwc <- emtrends(model, pairwise ~ treat, var = "time", adjust = "tukey")
contrasts <- data.frame(pwc$contrasts)
contrasts$p.signif <- symnum(contrasts$p.value, corr=FALSE, na=FALSE, legend=FALSE,
                             cutpoints = c(0, 0.001, 0.01, 0.05, 0.1, 1), 
                             symbols = c("***", "**", "*", ".", "ns"))
contrasts$grouped <- contrasts$contrast
contrasts <- contrasts %>% separate(c("grouped"), c("group1", "group2", "group3"), sep = " - ")
contrasts$p.value <- formatC(contrasts$p.value, format="e", digits=2)
slopes <- data.frame(pwc$emtrends) 
### bar plot
                                        #slopes$treat <- factor(slopes$treat, levels = c("0", "1"))
slopes$treat <- factor(slopes$treat, levels = c("pbs", "htt_aso", "scr_aso"))

diictrl.bar <- 
ggplot(slopes, aes(x=treat, y=time.trend)) +
  geom_bar(aes(fill=treat), stat="identity", color="black", position=position_dodge()) +
  geom_errorbar(aes(ymin=time.trend-SE, ymax=time.trend+SE), width=.2, position=position_dodge(.9)) +
  stat_pvalue_manual(contrasts, label = "{p.signif} \n p = {p.value}", y.position = 0.005, step.increase = 0.1, vjust=-0.1) +
#  scale_y_continuous(limits = c(NA, 0.006)) +
    scale_x_discrete(labels = c("PBS", expression(paste(italic("HTT"), " ASO")), expression(paste(italic("SCR"), " ASO")))) +
    scale_fill_discrete(labels=c("PBS", expression(paste(italic("HTT"), " ASO")), expression(paste(italic("SCR"), " ASO")))) +
#    scale_x_discrete(labels = c("FAN1-/-", "WT") )+
#    scale_fill_discrete(labels=c("FAN1-/-", "WT")) +

  # scale_colour_discrete(labels=c("PBS", "HTT ASO")) +
  geom_hline(yintercept = 0) +
  theme_bw() +
  labs(#title="genimix",
       x="Type", y="genimix (CAG/d)",
       fill="Type", colour="Type") +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size=15), 
        plot.subtitle = element_text(hjust = 0.5),
        text = element_text(size=15),
        plot.caption = element_text(),
        legend.text.align = 0
        # legend.text = element_text(margin = margin(0, 50, 0, 0))
        # axis.text.x = element_text(angle = 45, hjust = 1)
  )
diictrl.bar

#print (summary(lm(ii ~ time + treat + group, data=data)))
ggplot(data, aes(x=time, y=ii, colour=treat, shape=treat)) +
    geom_point() +
    geom_errorbar(aes(ymin=ii-ii_std, ymax=ii+ii_std), width=.2,
                  position=position_dodge(0.05)) +
      labs(
       x="time", y="iictrl") +

    geom_smooth(method='lm')

model <- lme(ii ~ time * treat, random = ~1 | id, na.action="na.omit", data=data)
anova(model)
pwc <- emtrends(model, pairwise ~ treat, var = "time", adjust = "tukey")
contrasts <- data.frame(pwc$contrasts)
contrasts$p.signif <- symnum(contrasts$p.value, corr=FALSE, na=FALSE, legend=FALSE,
                             cutpoints = c(0, 0.001, 0.01, 0.05, 0.1, 1), 
                             symbols = c("***", "**", "*", ".", "ns"))
contrasts$grouped <- contrasts$contrast
contrasts <- contrasts %>% separate(c("grouped"), c("group1", "group2"), sep = " - ")
contrasts$p.value <- formatC(contrasts$p.value, format="e", digits=2)
slopes <- data.frame(pwc$emtrends) 
### bar plot
                                        #slopes$treat <- factor(slopes$treat, levels = c("0", "1"))
slopes$treat <- factor(slopes$treat, levels = c("pbs", "htt_aso", "scr_aso"))

diictrl.bar <- 
ggplot(slopes, aes(x=treat, y=time.trend)) +
  geom_bar(aes(fill=treat), stat="identity", color="black", position=position_dodge()) +
  geom_errorbar(aes(ymin=time.trend-SE, ymax=time.trend+SE), width=.2, position=position_dodge(.9)) +
  stat_pvalue_manual(contrasts, label = "{p.signif} \n p = {p.value}", y.position = 0.005, step.increase = 0.1, vjust=-0.1) +
                                        #  scale_y_continuous(limits = c(NA, 0.006)) +
    scale_x_discrete(labels = c("PBS", expression(paste(italic("HTT"), " ASO")), expression(paste(italic("SCR"), " ASO")))) +
    scale_fill_discrete(labels=c("PBS", expression(paste(italic("HTT"), " ASO")), expression(paste(italic("SCR"), " ASO")))) +
#    scale_x_discrete(labels = c("FAN1-/-", "WT")) +
#    scale_fill_discrete(labels=c("FAN1-/-", "WT")) +

  # scale_colour_discrete(labels=c("PBS", "HTT ASO")) +
  geom_hline(yintercept = 0) +
  theme_bw() +
  labs(#title="iictrl",
       x="Type", y="iictrl (CAG/d)",
       fill="Type", colour="Type") +
  theme(plot.title = element_text(hjust = 0.5, face = "bold", size=15), 
        plot.subtitle = element_text(hjust = 0.5),
        text = element_text(size=15),
        plot.caption = element_text(),
        legend.text.align = 0
        # legend.text = element_text(margin = margin(0, 50, 0, 0))
        # axis.text.x = element_text(angle = 45, hjust = 1)
  )
diictrl.bar


#red <- data.frame(ladder = c(50, 75, 100, 125, 150, 200, 250, 300, 350, 400, 450, 475, 500, 
#                            550, 600, 650, 700, 750, 800, 850, 900, 950, 1000), 
#                 time = c(1564, 1812, 2060, 2308, 2562, 3066, 3582, 4097, 4612, 5131, 
#                          5639, 5879, 6127, 6610, 7076, 7524, 7946, 8346, 8713, 9050, 9357, 
#                          9630, 9871))

#print (summary(lm(ladder ~ poly(time,2), data=red)))
#print (summary(lm(ladder ~ poly(time,3), data=red)))

#ggplot(red, aes(x=time, y=ladder)) +
#  geom_point() +
#  geom_smooth(se=FALSE, method="loess", span=0.5)
