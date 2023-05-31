GRAPH=uk2007
GRAPHS="gsh-2015 clueweb12 uk-2014-csgr"
#SCHEME=streamvbyte
#SCHEME=cgr
SCHEME=hybrid
#FORMAT=dag-zeta2
FORMAT=dag-hybrid-32
BINDIR="/home/x-xhchen/proj/GraphAIBench/bin"
BIN="tc_omp_compressed"
OUTDIR="/home/x-xhchen/proj/outputs"

for GRAPH in $GRAPHS; do
  echo "$BINDIR/$BIN -s $SCHEME -i ~/data/$GRAPH/$FORMAT -o &> $OUTDIR/$BIN-$SCHEME-$GRAPH-$FORMAT.log"
  $BINDIR/$BIN -s $SCHEME -i ~/data/$GRAPH/$FORMAT -o &> $OUTDIR/$BIN-$SCHEME-$GRAPH-$FORMAT.log
done
