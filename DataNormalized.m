function [ x ] = DataNormalized( I )
%%%%%%%ʹһ�����ݹ�һ��


    MinValue = min(I);
    MaxValue = max(I);

    x = (I - MinValue)./(MaxValue - MinValue);
   

end

