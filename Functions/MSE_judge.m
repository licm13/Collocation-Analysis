function [data,omega] = MSE_judge(data,omega)
    if isnan(data)
        data = 0;
        omega = 0;
    elseif data<0
        data = 0;
        omega = 0;
    elseif isnan(omega)
        omega = 0;
        data = 0;
    elseif omega<0
        omega = 0;
        data = 0;
    end
end