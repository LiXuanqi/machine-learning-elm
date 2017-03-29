function [ x ] = DataNormalized( I )
%%%%%%%使一列数据归一化


    MinValue = min(I);
    MaxValue = max(I);

    x = (I - MinValue)./(MaxValue - MinValue);
   

end

