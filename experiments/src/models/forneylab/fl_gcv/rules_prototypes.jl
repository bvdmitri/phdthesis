@structuredVariationalRule(:node_type     => GaussianMeanPrecision,
                           :outbound_type => Message{Gaussian},
                           :inbound_types => (Nothing,Message{Function},ProbabilityDistribution),
                           :name          => SVBGaussianMeanPrecisionOutNED)

@structuredVariationalRule(:node_type => GaussianMeanPrecision,
                       :outbound_type => Message{GaussianMeanVariance},
                       :inbound_types => (Message{Function},Nothing,ProbabilityDistribution),
                       :name          => SVBGaussianMeanPrecisionMEND)

@marginalRule(:node_type     => GaussianMeanPrecision,
              :inbound_types => (Message{Gaussian}, Message{Function}, ProbabilityDistribution),
              :name          => MGaussianMeanPrecisionGED)

@marginalRule(:node_type     => GaussianMeanPrecision,
              :inbound_types => (Message{Function}, Message{Gaussian}, ProbabilityDistribution),
              :name          => MGaussianMeanPrecisionEGD)




@structuredVariationalRule(:node_type     => GaussianControlledVariance,
                           :outbound_type => Message{GaussianMeanVariance},
                           :inbound_types => (Nothing, Message{Gaussian}, ProbabilityDistribution, ProbabilityDistribution,ProbabilityDistribution),
                           :name          => SVBGaussianControlledVarianceOutNGDDD)

@structuredVariationalRule(:node_type     => GaussianControlledVariance,
                           :outbound_type => Message{GaussianMeanVariance},
                           :inbound_types => (Message{Gaussian},Nothing,ProbabilityDistribution, ProbabilityDistribution,ProbabilityDistribution),
                           :name          => SVBGaussianControlledVarianceXGNDDD)

@structuredVariationalRule(:node_type     => GaussianControlledVariance,
                           :outbound_type => Message{Function},
                           :inbound_types => (ProbabilityDistribution,Nothing,ProbabilityDistribution,ProbabilityDistribution),
                           :name          => SVBGaussianControlledVarianceZDNDD)

@structuredVariationalRule(:node_type     => GaussianControlledVariance,
                           :outbound_type => Message{Function},
                           :inbound_types => (ProbabilityDistribution,ProbabilityDistribution,Nothing,ProbabilityDistribution),
                           :name          => SVBGaussianControlledVarianceΚDDND)

@structuredVariationalRule(:node_type     => GaussianControlledVariance,
                           :outbound_type => Message{Function},
                           :inbound_types => (ProbabilityDistribution,ProbabilityDistribution,ProbabilityDistribution,Nothing),
                           :name          => SVBGaussianControlledVarianceΩDDDN)

@marginalRule(:node_type     => GaussianControlledVariance,
              :inbound_types => (Message{Gaussian},Message{Gaussian},ProbabilityDistribution,ProbabilityDistribution,ProbabilityDistribution),
              :name          => MGaussianControlledVarianceGGDDD)

#
@structuredVariationalRule(:node_type  => GaussianControlledVariance,
                     :outbound_type    => Message{GaussianMeanVariance},
                     :inbound_types    => (ProbabilityDistribution,Nothing,Message{Gaussian},ProbabilityDistribution),
                     :name             => SVBGaussianControlledVarianceZDGGD)
#
@structuredVariationalRule(:node_type     => GaussianControlledVariance,
                           :outbound_type => Message{GaussianMeanVariance},
                           :inbound_types => (ProbabilityDistribution,Message{Gaussian},Nothing,ProbabilityDistribution),
                           :name          => SVBGaussianControlledVarianceΚDGGD)

@structuredVariationalRule(:node_type   => GaussianControlledVariance,
                         :outbound_type => Message{GaussianMeanVariance},
                         :inbound_types => (Nothing, Message{Function}, ProbabilityDistribution, ProbabilityDistribution,ProbabilityDistribution),
                         :name          => SVBGaussianControlledVarianceOutNEDDD)

@structuredVariationalRule(:node_type    => GaussianControlledVariance,
                          :outbound_type => Message{GaussianMeanVariance},
                          :inbound_types => (Message{Function}, Nothing, ProbabilityDistribution, ProbabilityDistribution,ProbabilityDistribution),
                          :name          => SVBGaussianControlledVarianceMENDDD)


@naiveVariationalRule(:node_type     => GaussianControlledVariance,
                      :outbound_type => Message{GaussianMeanVariance},
                      :inbound_types => (Nothing, ProbabilityDistribution, ProbabilityDistribution,ProbabilityDistribution,ProbabilityDistribution),
                      :name          => VBGaussianControlledVarianceOutNDDDD)

@naiveVariationalRule(:node_type   => GaussianControlledVariance,
                    :outbound_type => Message{GaussianMeanVariance},
                    :inbound_types => (ProbabilityDistribution, Nothing,ProbabilityDistribution,ProbabilityDistribution,ProbabilityDistribution),
                    :name          => VBGaussianControlledVarianceXDNDDD)

@naiveVariationalRule(:node_type   => GaussianControlledVariance,
                    :outbound_type => Message{Function},
                    :inbound_types => (ProbabilityDistribution,ProbabilityDistribution,Nothing,ProbabilityDistribution,ProbabilityDistribution),
                    :name          => VBGaussianControlledVarianceZDDNDD)

@naiveVariationalRule(:node_type   => GaussianControlledVariance,
                    :outbound_type => Message{Function},
                    :inbound_types => (ProbabilityDistribution,ProbabilityDistribution,ProbabilityDistribution,Nothing,ProbabilityDistribution),
                    :name          => VBGaussianControlledVarianceKDDDND)

@naiveVariationalRule(:node_type   => GaussianControlledVariance,
                    :outbound_type => Message{Function},
                    :inbound_types => (ProbabilityDistribution,ProbabilityDistribution,ProbabilityDistribution,ProbabilityDistribution,Nothing),
                    :name          => VBGaussianControlledVarianceΩDDDDN)
