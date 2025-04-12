#set page(
  width: 14cm,
)
$
  M a = -gradient_x E(x)+f\
$
$
  M(delta t^(-2)delta x_(n+1)-delta t^(-2)delta x_n) = -gradient_x E(x_(n+1))-gradient_x^2E(x_(n+1))delta x_(n+1) + f\ 
$
$
  (delta t^(-2)M+gradient^2_x E(x_(n+1))) delta x_(n+1)=-gradient_x E(x_(n+1))+f+M delta t^(-2)delta x_n\
$
$
  E(x) = 2^(-1)||A x-b(x)||_2^2
$
$
  (delta t^(-2)M+A^top A) delta x_(n+1)=-A^top (A x_(n+1)-b(x_(n+1)))+f+M delta t^(-2) delta x_n\
$
differentiation:
$
  M a = -gradient_x E(x)+f\
$
$
  M(delta t^(-2) delta x_(n+1)-delta t^(-2) delta x_n) = -gradient_x E(x_(n+1)) + f\ 
$
$
  M delta t^(-2) gradient_x_n delta x_(n+1) = -gradient_x^2 E(x_(n+1)) gradient_x_n x_(n+1)\ 
  M delta t^(-2) gradient_x_n x_(n+1) - M delta t^(-2) I = -gradient_x^2 E(x_(n+1)) gradient_x_n x_(n+1)\ 
  (M delta t^(-2) + gradient_x^2 E(x_(n+1))) gradient_x_n x_(n+1) = M delta t^(-2) I\ 
$
$
  M (delta t^(-2) gradient_(delta x_n) delta x_(n+1) - delta t^(-2) I) = -gradient_x^2 E(x_(n+1)) gradient_(delta x_n) x_(n+1)\ 
  M delta t^(-2) gradient_(delta x_n) delta x_(n+1) - M delta t^(-2) I = -gradient_x^2 E(x_(n+1)) gradient_(delta x_n) delta x_(n+1)\ 
  (M delta t^(-2) + gradient_x^2 E(x_(n+1))) gradient_(delta x_n) delta x_(n+1) = M delta t^(-2) I\ 
$
$
  M (delta t^(-2) gradient_(delta x_n) delta x_(n+1) - t^(-2) I) = -gradient_x^2 E(x_(n+1)) gradient_(delta x_n) x_(n+1)\ 
  M delta t^(-2) gradient_(delta x_n) x_(n+1) - M delta t^(-2) I = -gradient_x^2 E(x_(n+1)) gradient_(delta x_n) x_(n+1)\ 
  (M delta t^(-2) + gradient_x^2 E(x_(n+1))) gradient_(delta x_n) x_(n+1) = M delta t^(-2) I\ 
$
$
  M delta t^(-2) gradient_x_n delta x_(n+1) = -gradient_x^2 E(x_(n+1)) gradient_x_n x_(n+1)\ 
  M delta t^(-2) gradient_x_n delta x_(n+1) = -gradient_x^2 E(x_(n+1)) gradient_x_n delta x_(n+1) - gradient_x^2 E(x_(n+1)) I \ 
  (M delta t^(-2) + gradient_x^2 E(x_(n+1))) gradient_x_n delta x_(n+1) = gradient_x^2 E(x_(n+1)) I\ 
$
$
  M delta t^(-2) gradient_b_(n+1) delta x_(n+1) = - gradient_(x,b) E(x_(n+1),b_(n+1))\ 
  M delta t^(-2) gradient_b_(n+1) x_(n+1) = - A^T A gradient_b_(n+1) x_(n+1) - A^T I\ 
  (M delta t^(-2) + A^T A) gradient_b_(n+1) x_(n+1) = - A^T I\ 
  (M delta t^(-2) + A^T A) gradient_b_(n+1) delta x_(n+1) = - A^T I\ 
$
$
  gradient_x^2 E(x) = A^T A - A^T gradient_x b(x)\
  gradient_(x,b) E(x,b) = A^T A gradient_b x + A^T I
$
$
  gradient_x_n L = gradient_x_(n+1) L gradient_x_n x_(n+1) + gradient_(delta x_(n+1)) L gradient_x_n delta x_(n+1)\ 
  gradient_(delta x_n) L = gradient_x_(n+1) L gradient_(delta x_n) x_(n+1) + gradient_(delta x_(n+1)) L gradient_(delta x_n) delta x_(n+1)\ 
$
$
  lambda := g (M delta t ^(-2) + A^T A - A^T gradient_x b(x))^(-1)\
  lambda (M delta t ^(-2) + A^T A - A^T gradient_x b(x)) = g\
  (M delta t ^(-2) + A^T A - A^T gradient_x b(x)) lambda = g\
  (M delta t ^(-2) + A^T A) lambda - A^T gradient_x b(x) lambda = g\
$
