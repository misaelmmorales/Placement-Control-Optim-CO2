function [VE_model,VE_wellSol,VE_states] = make_simulation(Gt, rock2D, VE_fluid, VE_initState, VE_schedule)
%MAKE_SIMULATION Summary of this function goes here

VE_model = CO2VEBlackOilTypeModel(Gt, rock2D, VE_fluid);
[VE_wellSol, VE_states] = simulateScheduleAD(VE_initState, VE_model, VE_schedule);
VE_states = [{VE_initState} VE_states(:)'];

end

