sample=$1
bam=$2
outdir=$(pwd)/$sample
mkdir $outdir
ref=/home/grads/gzpan2/ref/hg38_hpv.fa
samtools=/usr/bin/samtools

/home/grads/gzpan2/apps/miniconda3/envs/cityu/bin/python /home/grads/gzpan2/scripts/readfermikit.py -b $bam -o $outdir/extract.fq.gz && \
/home/grads/gzpan2/apps/fermikit/fermi.kit/fermi2.pl unitig -s3g -t8 -l 70 -p $outdir/$sample $outdir/extract.fq.gz >> $outdir/$sample.mak && \
make -f $outdir/$sample.mak && \
/home/grads/gzpan2/apps/fermikit/fermi.kit/run-calling -t8 $ref $outdir/$sample.mag.gz |sh && \
/home/grads/gzpan2/apps/fermikit/fermi.kit/htsbox abreak -l 70 -d 1 -p -c -f $ref $outdir/$sample.unsrt.sam.gz >$outdir/$sample.sv.2.vcf && \
less $outdir/$sample.sv.2.vcf |grep -v "##" | grep HPV >$outdir/virus_integrated.txt 
