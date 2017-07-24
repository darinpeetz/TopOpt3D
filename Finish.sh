n=`ls loads[0-9]* | wc | awk {'print $2'}`

if ((n==0))
then
  echo $n
  exit 0
fi

rm -f elements.bin nodes.bin supportNodes.bin supports.bin
rm -f springNodes.bin springs.bin loadNodes.bin loads.bin
rm -f edges.bin edgeLengths.bin massNodes.bin masses.bin

for ((i=0; i<$n; i++))
do
cat elements${i}.bin >> elements.bin
cat nodes${i}.bin >> nodes.bin
cat supportNodes${i}.bin >> supportNodes.bin
cat supports${i}.bin >> supports.bin
cat springNodes${i}.bin >> springNodes.bin
cat springs${i}.bin >> springs.bin
cat loadNodes${i}.bin >> loadNodes.bin
cat loads${i}.bin >> loads.bin
cat edges${i}.bin >> edges.bin
cat edgeLengths${i}.bin >> edgeLengths.bin
cat massNodes${i}.bin >> massNodes.bin
cat masses${i}.bin >> masses.bin
rm elements${i}.bin
rm nodes${i}.bin
rm supportNodes${i}.bin
rm supports${i}.bin
rm springNodes${i}.bin
rm springs${i}.bin
rm loadNodes${i}.bin
rm loads${i}.bin
rm edges${i}.bin
rm edgeLengths${i}.bin
rm massNodes${i}.bin
rm masses${i}.bin
done
