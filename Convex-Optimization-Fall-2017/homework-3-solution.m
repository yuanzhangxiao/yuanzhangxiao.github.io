% EE 617, Fall 2017, Homework 3

% generate problem data
m = 30;
n = 100;
A = randn( m , n ) + 1i * randn( m , n );
b = randn( m , 1 ) + 1i * randn( m , 1 );

% solve the problem directly
cvx_begin
    variable x(n) complex;
    minimize norm( x , inf )
    subject to 
        A * x == b
cvx_end

optimal_value_complex = cvx_optval;

% solve the SOCP with real variables
A_tilde = [ real(A) -imag(A) ; imag(A) real(A) ];
b_tilde = [ real(b) ; imag(b) ];

C = zeros( 2 , 2*n , n );

for i=1:1:n
    C(:,:,i) = zeros( 2 , 2*n );
    C( 1 , i , i ) = 1;
    C( 2 , n+i , i ) = 1;
end

cvx_begin
    variables t z(2*n);
    minimize t
    subject to 
        for i=1:1:n
            norm( C(:,:,i) * z , 2 ) <= t;
        end
        A_tilde * z == b_tilde;
cvx_end

optimal_value_real = cvx_optval;