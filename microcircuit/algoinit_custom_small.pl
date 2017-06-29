%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Copyright: TUD/UHEI 2007 - 2013
% License: GPL
% Description: BrainScaleS Mapping Process algorithm strategies init file
%              using GMPath [1].
%
%              Do not modify this file but use a user defined algoinit.pl
%              instead.
%
%
% [1] K. Wendt, M. Ehrlich, and R. Schüffny, "GMPath - a path language for 
%     navigation, information query and modification of data graphs", 
%     Proceedings of ANNIIP 2010, 2010, p. 31-42
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

EnableCreateMode

% get the global parameter node
GP = SystemNode/GlobalParameters

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Algorithm Sequence: user
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ASB = GP/AlgorithmSequences/user

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% System Algorithms Configuration
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

SystemAS = ASB/SystemAlgorithmSequence

% Linear Neuron Placement
A_PlacementNeurons = SystemAS/PlacementNeuronsSimple

% Recursive Mapping
A_RMS = SystemAS/RecursiveMappingStep

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% System Algorithm Sequence
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

SystemAS > FOLLOWER > A_PlacementNeurons > FOLLOWER > A_RMS

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Wafer Algorithms Placements
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

WaferAS = ASB/WaferAlgorithmSequence

% Linear Neuron Placement considering SynType to Abstract Neuron Slots
A_PlaceNeurons = WaferAS/PlacementNeuronsSimple
  % linearly select the hw componets to place to
  C = A_PlaceNeurons/RandomizeHWComponentSampling/0
  A_PlaceNeurons/RandomizeHWComponentSampling > EQUAL > C
  % linearly select the bio neuron during placment 
  C = A_PlaceNeurons/RandomizeBioNeuronSampling/1
  A_PlaceNeurons/RandomizeBioNeuornSampling > EQUAL > C
  % fill blocks to capacity?
  C = A_PlaceNeurons/FillBlocksToCap/0
  A_PlaceNeurons/FillBlocksToCap > EQUAL > C
  % consider the syntype when placing
  C = A_PlaceNeurons/SynType/1
  A_PlaceNeurons/SynType > EQUAL > C
  % place neurons directly to HICANN
  C = A_PlaceNeurons/Target/Invalid
  A_PlaceNeurons/Target > EQUAL > C
  % map poisson sources ?
  C = A_PlaceNeurons/MapPoissonSources/0
  A_PlaceNeurons/MapPoissonSources > EQUAL > C
  
% Hicann Out Pin Assignment
A_PlaceHicannOutPins = WaferAS/PlacementHicannOutPinAssignment
  % Target to which to map the elements to?
  C = A_PlaceHicannOutPins/Target/Invalid
  A_PlaceHicannOutPins/Target > EQUAL > C
  % shuffle bio neurons mapped on a hicann before assigning them to output registers and neuron slots.
  C = A_PlaceHicannOutPins/ShuffleBioNeurons/1
  A_PlaceHicannOutPins/ShuffleBioNeurons > EQUAL > C

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  Wafer Algorithms Routing
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Routing Layer 2 Routing 
% not taking into account L2 pulse rates
A_RouteL2 = WaferAS/RoutingL2RateDependent
  % randomize l1 addresses for virtual neurons
  C =  A_RouteL2/RandomL1Addresses/0
  A_RouteL2/RandomL1Addresses > EQUAL > C
  % Fraction of L2 bandwidth to utilize per DNC-HICANN link
  C =  A_RouteL2/AverageUtilization/0.8
  A_RouteL2/AverageUtilization > EQUAL > C

% Routing Layer 1  
% supported only for hardware model : FACETSModelV2
A_RouteL1 = WaferAS/RoutingL1WaferV2
  % enable one background event generator per used L1 bus
  % can be disabled when using the ESS
  C = A_RouteL1/OneBEGPerBus/1
  A_RouteL1/OneBEGPerBus > EQUAL > C

  % During CrossbarRouting:
  % if true, first expand all signals horizontally (i.e. create the horizontal backbone)
  %    then do the vertical expansion for all signals
  % if false, expand each Signal completely (horizontal and vertical) - one after another
  C = A_RouteL1/AllHorizontalFirst/1
  A_RouteL1/AllHorizontalFirst > EQUAL > C

  % During CrossbarRouting:
  % if true, during horizontal expansion, when you need to go vertical for routing around occupied elements,
  %    do the full vertical expansions in that column (considers both hicann columns)
  C = A_RouteL1/IfVerticalThenAll/1
  A_RouteL1/IfVerticalThenAll > EQUAL > C

  % During CrossbarRouting:
  % if true, consider mapping priority of synapses.
  C = A_RouteL1/ConsiderMappingPriority/1
  A_RouteL1/ConsiderMappingPriority > EQUAL > C

  % automatically adapt the number of hw-neurons if possible
  C =  A_RouteL1/AutoscaleNeuronSize/0
  A_RouteL1/AutoscaleNeuronSize > EQUAL > C
  % minimize absolut synapse loss : 1
  % balance synapse loss : 0
  C =  A_RouteL1/MinimizeSynapseLoss/0
  A_RouteL1/MinimizeSynapseLoss > EQUAL > C
  % fractrion of synapse driver that are used per HICANN (range: [0.,1])
  C =  A_RouteL1/SyndriverUtilization/0.9
  A_RouteL1/SyndriverUtilization > EQUAL > C

% Checks the validity of the routing config of the SPL1 Merger
% can be run after RoutingL1WaferV2
A_RouteL1Checker = WaferAS/RoutingSPL1MergerChecker

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  Wafer Algorithms Parameter Transformation
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Custom Parameter Transformation
A_ParamTrafo = WaferAS/ParameterTransformation
  % mode for transforming synaptic weights,
  % possible modes (bio->hardware)
  % [max2max,mean2mean,max2mean]
  C = A_ParamTrafo/WeightTrafoMode/max2max
  A_ParamTrafo/WeightTrafoMode > EQUAL > C
  % choose ideal transformation although
  % calibration data is available 
  C = A_ParamTrafo/IdealTransformation/1
  A_ParamTrafo/IdealTransformation > EQUAL > C
  % choose mode here auto_scale or fixed scaling
  % default is NeuronTrafoUseAutoScale<bool>(false)
  C = A_ParamTrafo/NeuronTrafoUseAutoScale/0
  A_ParamTrafo/NeuronTrafoUseAutoScale > EQUAL > C
  % parameter for NeuronTrafoUseAutoScale<bool>(true),
  % hw-range in mV between minimal V-reset and max V-thresh
  C = A_ParamTrafo/NeuronTrafoAutoScaleHWRange/200.
  A_ParamTrafo/A_ParamTrafo > EQUAL > C
  % parameter for NeuronTrafoUseAutoScale<bool>(true),
  % target reset voltage in mV
  C = A_ParamTrafo/NeuronTrafoAutoScaleTargetVReset/500.
  A_ParamTrafo/NeuronTrafoAutoScaleTargetVReset > EQUAL > C
  % parameter for NeuronTrafoUseAutoScale<bool>(false),
  % voltage shift factor in mV
  C = A_ParamTrafo/NeuronTrafoFixedScaleVShift/1200.
  A_ParamTrafo/NeuronTrafoFixedScaleVShift > EQUAL > C
  % parameter for NeuronTrafoUseAutoScale<bool>(false),
  % voltage scale factor
  C = A_ParamTrafo/NeuronTrafoFixedScaleAlphaV/10.
  A_ParamTrafo/NeuronTrafoFixedScaleAlphaV > EQUAL > C

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Wafer Algorithm Sequence
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

WaferAS > FOLLOWER > A_PlaceNeurons > FOLLOWER > A_PlaceHicannOutPins > FOLLOWER > A_RouteL2 > FOLLOWER > A_RouteL1 > FOLLOWER > A_RouteL1Checker > FOLLOWER > A_ParamTrafo

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Hicann Algorithm Sequence (currently not used)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

HicannAS = ASB/HicannAlgorithmSequence
