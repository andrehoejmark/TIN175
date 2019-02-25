
if test $2 ; then
  start_date="--start $2"
else
  start_date=""
fi
python3 ../src/utils/compactCSV.py --multi wind --folder Barkåkra --folder Göteborg --folder Jönköping --folder Väderöarna --folder Vinga temperature wind pressure --city barkakra --city gothenburg --city jonkoping --city vaderoarna --city vinga $start_date
