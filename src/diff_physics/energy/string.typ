#set page(
  width: 14cm,
)
projection:
$
  x_a - x_b = l (x_a - x_b)||x_a - x_b||_2^(-1)
$
$
  gradient_x_a l (x_a - x_b)||x_a - x_b||_2^(-1) =\
   l I ||x_a - x_b||_2^(-1) - l (x_a - x_b) ||x_a - x_b||_2^(-2)(x_a - x_b)^T I ||x_a - x_b||_2^(-1)\
   = l ||x_a - x_b||_2^(-1)(I - ||x_a - x_b||_2^(-2) (x_a - x_b) (x_a - x_b)^T)
$