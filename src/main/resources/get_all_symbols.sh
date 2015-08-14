mkdir -p data
while read SYMBOL; do
  ./get_symbol.sh ${SYMBOL}
  sleep 1
done < symbols.txt
