echo "Final evaluation for k = 1:"
grep "Final" full-evaluation-1.log | tail -n 1
echo "Final evaluation for k = 5:"
grep "Final" full-evaluation-k-5.log | tail -n 2
echo "Final evaluation for k = 10:"
grep "Final" full-evaluation-k-10.log | tail -n 2
echo "Final evaluation for k = 100:"
grep "Final" full-evaluation-k-100.log | tail -n 2
