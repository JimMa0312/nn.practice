clf();
i=2;
t=read('E:\\code\\Java\\AI\nn.practice\\src\\main\\resources\\FunctionCorrdinate\\x_4_1.0E-4.txt',1,100);
t1=read('E:\\code\\Java\\AI\nn.practice\\src\\main\\resources\\FunctionCorrdinate\\y_4_1.0E-4.txt',1,100);
x=[-2:0.04:2];
plot(t,t1,'k+',x,sin(%pi*i/4*x)+1,'k');
