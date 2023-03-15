library(ggplot2)
library(ggpubr)

pktsBytes <- readr::read_csv("capture20110810_pkts_bytes.csv", show_col_types=FALSE)

shapiro.test(pktsBytes$SrcPkts[1:5000])
#Shapiro-Wilk normality test
#
#data:  pktsBytes$SrcPkts[1:5000]
#W = 0.016447, p-value < 2.2e-16

# NO PUEDO ASUMIR NORMALIDAD (debería ser p-value > 0.005)
# ENTONCES NO PUEDO USAR CORRELACION DE PEARSON
# EL METODO "kendall" ESTUVO MAS DE 40 MINUTOS Y LO CORTE
# PRUEBO ENTONCES CON METODO "spearman"


res2 <-cor.test(pktsBytes$SrcPkts, pktsBytes$SrcBytes,  method = "spearman")
#Warning message:
#In cor.test.default(pktsBytes$SrcPkts, pktsBytes$SrcBytes, method = "spearman") :
#  Cannot compute exact p-value with ties

res2
#
#	Spearman's rank correlation rho
#
#data:  pktsBytes$SrcPkts and pktsBytes$SrcBytes
#S = 5.7718e+17, p-value < 2.2e-16
#alternative hypothesis: true rho is not equal to 0
#sample estimates:
#      rho 
#0.8384016 

ggscatter(pktsBytes[,c(3,5)], x="SrcPkts", y="SrcBytes", add="reg.line", conf.int=TRUE, cor.coef=TRUE, cor.method="spearman")
#`geom_smooth()` using formula 'y ~ x'

ggsave("capture20110810_Src_correlation_spearman.png")
#Saving 6.99 x 6.99 in image
#`geom_smooth()` using formula 'y ~ x'


############################3

resDst <-cor.test(pktsBytes$DstPkts, pktsBytes$DstBytes,  method = "spearman") 
#Warning message:
#In cor.test.default(pktsBytes$DstPkts, pktsBytes$DstBytes, method = "spearman") :
#  Cannot compute exact p-value with ties

resDst
#
#	Spearman's rank correlation rho
#
#data:  pktsBytes$DstPkts and pktsBytes$DstBytes
#S = 1.2367e+18, p-value < 2.2e-16
#alternative hypothesis: true rho is not equal to 0
#sample estimates:
#      rho 
#0.6537583 

ggscatter(pktsBytes[,c(3,5)], x="SrcPkts", y="SrcBytes", add="reg.line", add.params=list(color="blue",fill="lightgray"), conf.int=TRUE, cor.coef=TRUE, cor.method="spearman")
#`geom_smooth()` using formula 'y ~ x'

ggsave("capture20110810_Src_correlation_spearman.png")
#Saving 7 x 7 in image
#`geom_smooth()` using formula 'y ~ x'

