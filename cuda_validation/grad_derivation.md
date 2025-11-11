
## Vanilla 1D Mamba

```cpp
const float dx = thread_reverse_data[i].y;

// VANILLA 1D MAMBA RECURSION
// h_t == exp(delta_t A_t) h_{t-1} + delta_t B_t u_t
// >>>>>> Derivation of ddelta_vals[i]
// pdv{Loss}{h_t} known == dx
// pdv{h_t}{delta_t} == h_{t-1} exp(delta_t A_t) A_t + B_t u_t
//                   == [ h_{t-1} exp(delta_t A_t) ] A_t + B_t u_t
//                   == [ h_t - B_t u_t ] A_t + B_t u_t
//                   == a * A_Val + ...
// ddelta_vals[i] == pdv{Loss}{delta_t}
//                == pdv{Loss}{delta_t} pdv{h_t}{delta_t}
//                == dx * pdv{h_t}{delta_t}
//                == ↓
ddelta_vals[i] +=
        ddelta_u * float(u_vals[i]) +  // <-- for delta's ops on B
        dx * A_val * a;                // <-- for delta's ops on A

// VANILLA 1D MAMBA RECURSION
// h_t == exp(delta_t A_t) h_{t-1} + delta_t B_t u_t
// >>>>>> Derivation of dA_val
// pdv{h_t}{A_t} == h_{t-1} exp(delta_t A_t) delta_t
//               == [ h_{t-1} exp(delta_t A_t) ] delta_t
//               == [ h_t - B_t u_t ] delta_t
//               == a * delta_vals[i]
// dA_val == pdv{Loss}{A_t}
//        == pdv{Loss}{h_t} pdv{h_t}{A_t}
//        == dx * a * delta_vals[i]
//        ==  ↓
dA_val += dx * delta_vals[i] * a;
```


## Locally Bidirectional Mamba

$ h_t^L = \exp(\Delta_t A_t) h_{t-1}^L + \Delta_t B_t u_t $

$ h_t^R = \exp(\Delta_t A_t) h_{t+1}^R + \Delta_t B_t u_t $

$ h_t = \underbrace{h_t^L}_{\texttt{thread\_data}} + \underbrace{h_t^R}_{\texttt{thread\_hr\_data}} - \Delta_t B_t u_t $


$ H = H^L + H^R + H^u $

- $ H = \left( H_t(h_t, h_{t-1}, \cdots, h_1), H_{t-1}(h_{t-1}, h_{t-2}, \cdots, h_1), \cdots, H_2(h_2, h_1), H_1(h_1) \right) $
  - $ t = 1, 2, \cdots, n $ 
  - Here $h$ and $H$ have superscripts $L$ or $R$

$ 
\dfrac{\partial F}{\partial H} = 
\dfrac{\partial F}{\partial H^L} = 
\dfrac{\partial F}{\partial H^R} = 
\dfrac{\partial F}{\partial H^u}
$

- Bwd kernel left scan
  - Before:
    - $ \mathtt{thread\_reverse\_data.x} = \exp(\Delta_{t+1} A_{t+1}) $ SHIFTED
    - $ \mathtt{thread\_reverse\_data.y} = \dfrac{\partial F}{\partial H^L} B $
  - After: 
    - $ \underbrace{\mathtt{thread\_reverse\_data[t].y} = \dfrac{\partial F}{\partial H^L} \dfrac{\partial H^L}{\partial h^L_t} = \dfrac{\partial F}{\partial h_t^L}}_{\texttt{dx\_left}} $
    - 
    - The scan applies $ \dfrac{\partial F}{\partial H^L} $
- Bwd kernel right scan
  - Before:
    - $ \mathtt{thread\_hr\_reverse\_data.x} = \exp(\Delta_{t-1} A_{t-1}) $ SHIFTED
    - $ \mathtt{thread\_hr\_reverse\_data.y} = \dfrac{\partial F}{\partial H^R} B $
  - After: 
    - $ \underbrace{\mathtt{thread\_hr\_reverse\_data[t].y} = \dfrac{\partial F}{\partial H^R} \dfrac{\partial H^R}{\partial h^R_t} = \dfrac{\partial F}{\partial h_t^R}}_{\texttt{dx\_right}} $
    - 
    - The scan applies $ \dfrac{\partial F}{\partial H^R} $


<!-- $ \dfrac{\partial F}{\partial h_t} \text{ known } = \mathtt{thread\_reverse\_data} $ -->

- `du_vals`
```cpp
du_vals[i] = (dx_left + dx_right - 1.0f) * B_vals[i] * delta_vals[i];
```
- `ddelta_vals`
  - \
$ 
\dfrac{\partial F}{\partial \Delta_t} = 
\dfrac{\partial F}{\partial h_t^L} \dfrac{\partial h_t^L}{\partial \Delta_t} +
\dfrac{\partial F}{\partial h_t^R} \dfrac{\partial h_t^R}{\partial \Delta_t} - 
\dfrac{\partial F}{\partial h_u} B_t u_t 
$
  - \ 
$
\dfrac{\partial F}{\partial \Delta_t} = 
\dfrac{\partial F}{\partial h_t^L} (\exp(\Delta_t A_t) h_{t-1}^L A_t + B_t u_t) +
\dfrac{\partial F}{\partial h_t^R} (\exp(\Delta_t A_t) h_{t+1}^R A_t + B_t u_t) - 
\dfrac{\partial F}{\partial h_u} B_t u_t 
$
  - \
$ 
\dfrac{\partial F}{\partial \Delta_t} = 
\dfrac{\partial F}{\partial h_t^L} ( \underbrace{( h_t^L - \Delta_t B_t u_t )}_{\texttt{a\_left}} A_t + B_t u_t ) + 
\dfrac{\partial F}{\partial h_t^R} ( \underbrace{( h_t^R - \Delta_t B_t u_t )}_{\texttt{a\_right}} A_t + B_t u_t ) - 
\dfrac{\partial F}{\partial h_u} B_t u_t
$
- `dA_vals`
  - \
$
\dfrac{\partial F}{\partial A_t} = 
\dfrac{\partial F}{\partial h_t^L} \dfrac{\partial h_t^L}{\partial A_t} + 
\dfrac{\partial F}{\partial h_t^L} \dfrac{\partial h_t^R}{\partial A_t} 
$
  - \
$
\dfrac{\partial F}{\partial A_t} = 
\dfrac{\partial F}{\partial h_t^L} ( \{ h_{t-1}^L \exp(\Delta_t A_t) \} \Delta_t ) + 
\dfrac{\partial F}{\partial h_t^R} ( \{ h_{t+1}^R \exp(\Delta_t A_t) \} \Delta_t ) 
$
  - \
$
\dfrac{\partial F}{\partial A_t} =  
\dfrac{\partial F}{\partial h_t^L} ( \underbrace{( h_t^L - \Delta_t B_t u_t )}_{\texttt{a\_left}} \Delta_t ) + 
\dfrac{\partial F}{\partial h_t^R} ( \underbrace{( h_t^R - \Delta_t B_t u_t )}_{\texttt{a\_right}} \Delta_t )
$
- `dB_vals`
  - \
$
\dfrac{\partial F}{\partial B_t} = 
\dfrac{\partial F}{\partial h_t^L} \dfrac{\partial h_t^L}{\partial B_t} + 
\dfrac{\partial F}{\partial h_t^R} \dfrac{\partial h_t^R}{\partial B_t} - 
\dfrac{\partial F}{\partial h_u} \Delta_t u_t
$
  - \
$
\dfrac{\partial F}{\partial B_t} = 
\dfrac{\partial F}{\partial h_t^L} \Delta_t u_t + 
\dfrac{\partial F}{\partial h_t^R} \Delta_t u_t - 
\dfrac{\partial F}{\partial h_u}\Delta_t u_t 
$
  - \
$
\dfrac{\partial F}{\partial B_t} =  
(\underbrace{\dfrac{\partial F}{\partial h_t^L}}_{\mathtt{dx\_left}} + 
 \underbrace{\dfrac{\partial F}{\partial h_t^R}}_{\mathtt{dx\_right}} - \dfrac{\partial F}{\partial h_u}) 
 \underbrace{\Delta_t u_t}_{\mathtt{delta\_vals[i] * u\_vals[i]}}
$
- `dC_vals`
  - $ y = \sum_{s} C_s H^{(s)} $
  - \
$ 
\dfrac{\partial F}{\partial C_s} = 
\dfrac{\partial F}{\partial y} H^{(s)} = 
\underbrace{\dfrac{\partial F}{\partial y}}_{\mathtt{dout\_vals[i]}} (H^L + H^R + H^u)
$
```cpp
if constexpr (kIsVariableC) {
    if constexpr (!kIsVariableB) {
        // constant
        const float H = thread_data[i].y + thread_hr_data[i].y - delta_vals[i] * float(u_vals[i]);
        dC_vals[i] = dout_vals[i] * H * B_val;
    } else {
        const float H = thread_data[i].y + thread_hr_data[i].y - delta_vals[i] * float(u_vals[i]) * B_vals[i];
        dC_vals[i] = dout_vals[i] * H;
    }
}
```