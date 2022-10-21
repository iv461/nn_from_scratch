Numpy notes:

- braodcasting expandiert die Achsen eines kleineren Vektors
- broadcasting passiert automatisch, z.B. bei Addition

a = np.arange(27).reshape(3, 3, 3)
np.sum(a) = 351
# starts axis counting from last, thus same as np.sum(a, axis=a.ndims-1) 
np.sum(a, axis=-1) = 
array([[ 3, 12, 21],
       [30, 39, 48],
       [57, 66, 75]])

np.sum(a, axis=-2) =

array([[ 9, 12, 15],
       [36, 39, 42],
       [63, 66, 69]])

np.sum(a, axis=-3) =

array([[27, 30, 33],
       [36, 39, 42],
       [45, 48, 51]])






