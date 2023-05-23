latin_square_2 = [[(i+j)%2 for i in range(2)] for j in range(2)]
print(latin_square_2)
latin_square_3 = [[(i+j)%3 for i in range(3)] for j in range(3)]
print(latin_square_3)

job = [0,1,2,3,4,5]

batch_2 = [job[2*i:2*i+2] for i in range(3)]
print(f'batch_2: {batch_2}')

latin_2_batches = []
for batch in batch_2:
    latin = [[batch[(i+j)%2] for i in range(2)] for j in range(2)]
    latin_2_batches.append(latin)

latin_6 = [[latin_2_batches[(i+j)%3] for i in range(3)] for j in range(3)]
print(latin_6)