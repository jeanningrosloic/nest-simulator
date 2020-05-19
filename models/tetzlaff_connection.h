/*
 *  stdp_connection.h
 *
 *  This file is part of NEST.
 *
 *  Copyright (C) 2004 The NEST Initiative
 *
 *  NEST is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  NEST is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with NEST.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef TETZLAFF_CONNECTION_H
#define TETZLAFF_CONNECTION_H

// C++ includes:
#include <cmath>

// Includes from nestkernel:
#include "common_synapse_properties.h"
#include "connection.h"
#include "connector_model.h"
#include "event.h"

// Includes from sli:
#include "dictdatum.h"
#include "dictutils.h"

namespace nest
{

/* BeginDocumentation
  Name: tetzlaff_synapse - Synapse type for spike-timing dependent
    plasticity with homeostatic scaling.
  Description:
    tetzlaff_synapse is a connector to create synapses with spike time
    dependent plasticity. Unlike stdp_synapse, we use the update equations:
    \Delta w = \lambda * ( x * y + \kappa * ( y_{tgt} - y ) * w^2 )
    where x and y are the pre- and postsynaptic traces (augmented by one when
    the neuron spikes and decaying excponentially wrt. tau_plus (pre) or
    tau_minus (post)), y_tgt is a constant postsynaptic target that scales
    postsynaptic activity, lambda and kappa are learning rates.
    Note that the weight updade above is performed every time step in its
    original formulation [1]. For this reason, the weight update equation that
    is performed at each spike emission can be written as if follows:
    \Delta w = \lambda * (
      ( kplus_1 * kminus_1 - kplus_2 * kminus_2 ) * \tau_{conv}
      + \kappa * w^2 * (
        kminus_{tgt} * ( t_2 - t_1 ) - ( kminus_1 - kminus_2 ) * \tau_-
      )
    )
    where:
      - \tau_{conv} = \tau_+ * \tau_- / ( \tau_+ + \tau_- )
      - index 1 describes states at the time of the previous/last spike
      - index 2 describes states at the current spike time
      - kplus/kminus are pre-/postsynaptic traces.
  Parameters:
    lambda          double - Step size / Learning rate
    Wmax            double - Maximum allowed weight, note that this scales each
                            weight update
    tau_plus        double - Time constant of the presynaptic trace (ms)
    tau_minus       double - Time constant of the postsynaptic trace (ms) note:
                             should be the same as set in the postsyn. neuron
    kappa           double - Homeostatic learning rate (regulates postsynaptic
                             scaling step size)
    Kminus_target   double - Homeostatic scaling level (determines the "desired"
                             postsyn. activity/firing rate)
  Transmits: SpikeEvent
  References:
    [1] Tetzlaff, C., Kolodziejski, C., Timme, M., Tsodyks, M., & Wörgötter,
        F. (2013). Synaptic scaling enables dynamically distinct short-and
        long-term memory formation. BMC neuroscience, 14(1), P415.
    [2] Tetzlaff, C., Kolodziejski, C., Timme, M., & Wörgötter, F. (2011).
        Synaptic scaling in combination with many generic plasticity mechanisms
        stabilizes circuit connectivity. Frontiers in computational
        neuroscience, 5, 47.
  Adapted from stdp_synapse:
      FirstVersion: March 2006
      Author: Moritz Helias, Abigail Morrison
      Adapted by: Philipp Weidel
  Author: Loïc Jeanningros (loic.jeanningros@gmail.com)
  SeeAlso: synapsedict, stdp_synapse
*/

// connections are templates of target identifier type (used for pointer /
// target index addressing) derived from generic connection template
template < typename targetidentifierT >
class TetzlaffConnection : public Connection< targetidentifierT >
{

public:
  typedef CommonSynapseProperties CommonPropertiesType;
  typedef Connection< targetidentifierT > ConnectionBase;

  /**
   * Default Constructor.
   * Sets default values for all parameters. Needed by GenericConnectorModel.
   */
  TetzlaffConnection();


  /**
   * Copy constructor.
   * Needs to be defined properly in order for GenericConnector to work.
   */
  TetzlaffConnection( const TetzlaffConnection& );

  // Explicitly declare all methods inherited from the dependent base
  // ConnectionBase. This avoids explicit name prefixes in all places these
  // functions are used. Since ConnectionBase depends on the template parameter,
  // they are not automatically found in the base class.
  using ConnectionBase::get_delay_steps;
  using ConnectionBase::get_delay;
  using ConnectionBase::get_rport;
  using ConnectionBase::get_target;

  /**
   * Get all properties of this connection and put them into a dictionary.
   */
  void get_status( DictionaryDatum& d ) const;

  /**
   * Set properties of this connection from the values given in dictionary.
   */
  void set_status( const DictionaryDatum& d, ConnectorModel& cm );

  /**
   * Send an event to the receiver of this connection.
   * \param e The event to send
   * \param cp common properties of all synapses (empty).
   */
  void send( Event& e, thread t, const CommonSynapseProperties& cp );


  class ConnTestDummyNode : public ConnTestDummyNodeBase
  {
  public:
    // Ensure proper overriding of overloaded virtual functions.
    // Return values from functions are ignored.
    using ConnTestDummyNodeBase::handles_test_event;
    port
    handles_test_event( SpikeEvent&, rport )
    {
      return invalid_port_;
    }
  };

  void
  check_connection( Node& s, Node& t, rport receptor_type, const CommonPropertiesType& )
  {
    ConnTestDummyNode dummy_target;

    ConnectionBase::check_connection_( dummy_target, s, t, receptor_type );

    t.register_stdp_connection( t_lastspike_ - get_delay(), get_delay() );
  }

  void
  set_weight( double w )
  {
    weight_ = w;
  }

private:
  double
  update_(
    double w
    , double kplus1
    , double kminus1
    , double kplus2
    , double kminus2
    , double minus_dt
    , double tau_conv_
  )
  {
    if(lambda_ == 0.0)
      return w;

    double dW = lambda_ * (
      ( kplus1*kminus1 - kplus2*kminus2 ) * tau_conv_
      + kappa_ * std::pow(w, 2) * (
        Kminus_target_ * ( - minus_dt ) - ( kminus1 - kminus2 ) * tau_minus_
      )
    );

    double new_w = w + dW;

    if( new_w < 0.0 )
      return 0.0;
    if (new_w > Wmax_ )
      return Wmax_;
    return new_w;

  }

  // data members of each connection
  double weight_;
  double tau_plus_;
  double tau_minus_;
  double lambda_;
  double Wmax_;
  double Kplus_;
  double kappa_;
  double Kminus_target_;

  double t_lastspike_;
};


/**
 * Send an event to the receiver of this connection.
 * \param e The event to send
 * \param t The thread on which this connection is stored.
 * \param cp Common properties object, containing the stdp parameters.
 */
template < typename targetidentifierT >
inline void
TetzlaffConnection< targetidentifierT >::send( Event& e, thread t, const CommonSynapseProperties& )
{
  // synapse STDP depressing/facilitation dynamics
  const double t_spike = e.get_stamp().get_ms();
  const double tau_conv_ = tau_plus_ * tau_minus_ / ( tau_plus_ + tau_minus_ );

  // use accessor functions (inherited from Connection< >) to obtain delay and
  // target
  Node* target = get_target( t );
  double dendritic_delay = get_delay();

  // get spike history in relevant range (t1, t2] from post-synaptic neuron
  std::deque< histentry >::iterator start;
  std::deque< histentry >::iterator finish;

  // For a new synapse, t_lastspike_ contains the point in time of the last
  // spike. So we initially read the
  // history(t_last_spike - dendritic_delay, ..., T_spike-dendritic_delay]
  // which increases the access counter for these entries.
  // At registration, all entries' access counters of
  // history[0, ..., t_last_spike - dendritic_delay] have been
  // incremented by Archiving_Node::register_stdp_connection(). See bug #218 for
  // details.
  target->get_history( t_lastspike_ - dendritic_delay, t_spike - dendritic_delay, &start, &finish );
  // weight update due to post-synaptic spikes since last pre-synaptic spike
  double minus_dt;
  double Kminus = target->get_K_value( t_lastspike_ - dendritic_delay );
  while ( start != finish )
  {
    minus_dt = t_lastspike_ - ( start->t_ + dendritic_delay );
    // get_history() should make sure that
    // start->t_ > t_lastspike - dendritic_delay, i.e. minus_dt < 0
    assert( minus_dt < -1.0 * kernel().connection_manager.get_stdp_eps() );
    // assert( minus_dt <= 0.0 );
    weight_ = update_(
      weight_
      , Kplus_
      , Kminus
      , Kplus_ * std::exp( minus_dt / tau_plus_ )
      , Kminus * std::exp( minus_dt / tau_minus_ )
      , minus_dt
      , tau_conv_
    );

    Kplus_ = Kplus_ * std::exp( minus_dt / tau_plus_ );
    Kminus = Kminus * std::exp( minus_dt / tau_minus_ ) + 1.0;

    std::cout << "N t_spk:" << start->t_ + dendritic_delay << std::endl;
    std::cout << "N Kminus: " << Kminus << std::endl;

    t_lastspike_ = ( start->t_ + dendritic_delay );
    ++start;
  }
  // weight update due to pre-synaptic spike
  minus_dt = t_lastspike_ - t_spike;

  weight_ = update_(
    weight_
    , Kplus_
    , Kminus
    , Kplus_ * std::exp( minus_dt / tau_plus_ )
    , Kminus * std::exp( minus_dt / tau_minus_ )
    , minus_dt
    , tau_conv_
  );

  e.set_receiver( *target );
  e.set_weight( weight_ );
  // use accessor functions (inherited from Connection< >) to obtain delay in
  // steps and rport
  e.set_delay_steps( get_delay_steps() );
  e.set_rport( get_rport() );
  e();

  Kplus_ = Kplus_ * std::exp( minus_dt / tau_plus_ ) + 1.0;

  std::cout << "N t_spk:" << t_spike << std::endl;
  std::cout << "N Kplus: " << Kplus_ << std::endl;

  t_lastspike_ = t_spike;
}


template < typename targetidentifierT >
TetzlaffConnection< targetidentifierT >::TetzlaffConnection()
  : ConnectionBase()
  , weight_( 1.0 )
  , tau_plus_( 20.0 )
  , tau_minus_( 20.0 )
  , lambda_( 0.01 )
  , Wmax_( 100.0 )
  , Kplus_( 0.0 )
  , kappa_( 0.01 )
  , Kminus_target_( 0.0 )
  , t_lastspike_( 0.0 )
{
}

template < typename targetidentifierT >
TetzlaffConnection< targetidentifierT >::TetzlaffConnection( const TetzlaffConnection< targetidentifierT >& rhs )
  : ConnectionBase( rhs )
  , weight_( rhs.weight_ )
  , tau_plus_( rhs.tau_plus_ )
  , tau_minus_( rhs.tau_minus_ )
  , lambda_( rhs.lambda_ )
  , Wmax_( rhs.Wmax_ )
  , Kplus_( rhs.Kplus_ )
  , kappa_( rhs.kappa_ )
  , Kminus_target_( rhs.Kminus_target_ )
  , t_lastspike_( rhs.t_lastspike_ )
{
}

template < typename targetidentifierT >
void
TetzlaffConnection< targetidentifierT >::get_status( DictionaryDatum& d ) const
{
  ConnectionBase::get_status( d );
  def< double >( d, names::weight, weight_ );
  def< double >( d, names::tau_plus, tau_plus_ );
  def< double >( d, names::tau_minus, tau_minus_ );
  def< double >( d, names::lambda, lambda_ );
  def< double >( d, names::Wmax, Wmax_ );
  def< double >( d, names::kappa, kappa_ );
  def< double >( d, names::Kminus_target, Kminus_target_ );
  def< long >( d, names::size_of, sizeof( *this ) );
}

template < typename targetidentifierT >
void
TetzlaffConnection< targetidentifierT >::set_status( const DictionaryDatum& d, ConnectorModel& cm )
{
  ConnectionBase::set_status( d, cm );
  updateValue< double >( d, names::weight, weight_ );
  updateValue< double >( d, names::tau_plus, tau_plus_ );
  updateValue< double >( d, names::tau_minus, tau_minus_ );
  updateValue< double >( d, names::lambda, lambda_ );
  updateValue< double >( d, names::Wmax, Wmax_ );
  updateValue< double >( d, names::kappa, kappa_ );
  updateValue< double >( d, names::Kminus_target, Kminus_target_ );

  // check if weight_ and Wmax_ has the same sign
  if ( not( ( ( weight_ >= 0 ) - ( weight_ < 0 ) ) == ( ( Wmax_ >= 0 ) - ( Wmax_ < 0 ) ) ) )
  {
    throw BadProperty( "Weight and Wmax must have same sign." );
  }
}

} // of namespace nest

#endif // of #ifndef TETZLAFF_CONNECTION_H
