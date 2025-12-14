#!/home/wangmengyao/anaconda3/bin/python
def read4fermikit(bamFile, outFile):
        n = 0
        samfile = pysam.AlignmentFile(bamFile)
        fout = gzip.open(outFile, 'wt')
        read = None
        for r in samfile.fetch(until_eof=True):
                if r.is_read1:
                        read = 1
                elif r.is_read2:
                        read = 2
                else:
                        read = 0
                if r.is_unmapped:
                        seq = r.query_sequence
                        qname = r.query_name
                        lout = "@%s#/%s\n%s\n+\n%s" % (qname, read, seq, r.qual)
                        print (lout, file=fout)
                        n += 1
                else:
                        cigarstring = r.cigarstring
                        try:
                                NM = r.get_tag('NM')
                        except:
                                NM = 0
                                pass
                        tid = r.reference_id
                        chrom = samfile.getrname(tid)
                        if 'S' in cigarstring or NM >= 5 or chrom.startswith('HPV'):
                                seq = r.query_sequence
                                qname = r.query_name
                                lout = "@%s#/%s\n%s\n+\n%s" % (qname, read, seq, r.qual)
                                print (lout, file=fout)
                                n += 1
        fout.close()
        samfile.close()
        print(sys.stderr, "Total reads extracted: %s" % n)

if __name__ == '__main__':
        import argparse
        parser = argparse.ArgumentParser(prog=None, description='Extract reads for fermikit assembly')
        parser.add_argument('-b', '--bamFile', dest='bamFile', help="input bam file")
        parser.add_argument('-o', '--outFile', dest='outFile', help="output file, ended with .gz")
        args = parser.parse_args()

        import pysam
        import gzip
        import sys
        read4fermikit(args.bamFile, args.outFile)

