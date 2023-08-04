echo "Current evaluation for k = 1:"
grep "Current" full-evaluation-1.log | tail -n 1
echo "Current evaluation for k = 5:"
grep "Current" full-evaluation-k-5.log | tail -n 2
echo "Current evaluation for k = 10:"
grep "Current" full-evaluation-k-10.log | tail -n 2
echo "Current evaluation for k = 100:"
grep "Current" full-evaluation-k-100.log | tail -n 2