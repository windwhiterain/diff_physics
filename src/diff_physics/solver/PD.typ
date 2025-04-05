#set page(
  width: 14cm,
)
$
M a = -gradient_x E(x)+f\
$
$
M(delta t_(n+1)^(-2)delta x_(n+1)-delta t_(n+1)^(-1)delta t_n^(-1)delta x_n) = -gradient_x E(x_n)-gradient_x^2E(x_n)delta x_(n+1) + f\ 
$
$
(delta t_(n+1)^(-2)M+gradient^2_x E(x_n)) delta x_(n+1)=-gradient_x E(x_n)+f+M delta t_(n+1)^(-1)delta t_n^(-1)delta x_n\
$
$
E(x) = 2^(-1)||A x-b||_2^2
$
$
(delta t_(n+1)^(-2)M+A^top A) delta x_(n+1)=-A^top (A x-b)+f+M delta t_(n+1)^(-1)delta t_n^(-1)delta x_n\
$