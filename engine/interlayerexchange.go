package engine

// Interlayer exchange coupling (IEC)
import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

// Anisotropy variables
var (
	J_iec1, J_iec2         	ScalarParam  	// bilinear and biquadratic coupling constants
	iec_top, iec_bottom		ScalarParam 	// top- and bottom layer positions
	J1_red, J2_red          derivedParam 	// 1e9 * J1 / Msat and 1e9 * J2 / Msat
	iec_direction	    	VectorParam 	// Direction of interlayer exchange
	B_iec                	vAdder     		// field due to IEC (T)
	E_iec                	*GetScalar 		// IEC energy
	Edens_iec            	sAdder     		// IEC density
	One 					int
)

func init() {
	J_iec1.init("J_iec1", "J/m2", "Bilinear IEC coupling constant", []derived{&J1_red})
	J_iec2.init("J_iec2", "J/m2", "Biquadratic IEC coupling constant", []derived{&J2_red})
	iec_top.init("IEC_top", "#", "Position of the top layer for IEC",[]derived{&top_red})
	iec_bottom.init("IEC_bottom", "#", "Position of the bottom layer for IEC",[]derived{&bottom_red})
	iec_direction.init("IEC_direction", "", "Direction of interlayer-exchange coupling")
	B_iec.init("B_iec", "T", "Interlayer-exchange coupling field", AddInterlayerField)
	E_iec = NewGetScalar("E_iec", "J", "Interlayer-exchange coupling energy", GetInterlayerEnergy)
	Edens_iec.init("Edens_iec", "J/m3", "Interlayer-exchange energy density", makeEdensAdder(&B_iec, -0.5))
	registerEnergy(GetInterlayerEnergy, Edens_iec.AddTo)

	// J1_red = 1e9 * J1 / Msat
	J1_red.init(1, []updater{&J_iec1, &Msat}, func(p *derivedParam) {
		paramDivMult(p.cpu_buf, J_iec1.cpuLUT(), Msat.cpuLUT(), 1e9)
	})

	// J2_red = 1e9 * J2 / Msat
	J2_red.init(1, []updater{&J_iec2, &Msat}, func(p *derivedParam) {
		paramDivMult(p.cpu_buf, J_iec2.cpuLUT(), Msat.cpuLUT(), 1e9)
	})

}

// Allows multiply by a constant onto paramDiv
func paramDivMult(dst, a, b [][NREGION]float32, c float32) {
	util.Assert(len(dst) == 1 && len(a) == 1 && len(b) == 1)
	for i := 0; i < NREGION; i++ { // not regions.maxreg
		dst[0][i] = c * safediv(a[0][i], b[0][i])
	}
}

// Add the interlayer exchange field to dst
func AddInterlayerField(dst *data.Slice) {
	if !(J1_red.isZero()) || !(J2_red.isZero()) {
		cuda.AddInterlayerExchange(
			dst, M.Buffer(), J1_red.gpuLUT1(), J2_red.gpuLUT1(), iec_top.gpuLUT1(),
			ec_bottom.gpuLUT1(), iec_direction.gpuLUT(), regions.Gpu(), Mesh()
		)
	}
}
/*func AddInterlayerEnergyDensity(dst *data.Slice) {
	buf := cuda.Buffer(B_iec.NComp(), B_iec.Mesh().Size())
	defer cuda.Recycle(buf)

	// unnormalized magnetization:
	Mf, r := M_full.Slice()
	if r {
		defer cuda.Recycle(Mf)
	}
	if !(J1_red.isZero()) || !(J2_red.isZero()) {
		cuda.Zero(buf)	
		cuda.AddInterlayerExchange(buf, M.Buffer(), J1_red.gpuLUT1(), J2_red.gpuLUT1(), iec_top.gpuLUT1(), iec_bottom.gpuLUT1(), regions.Gpu(), Mesh())
		cuda.AddDotProduct(dst, -1./2., buf, Mf)
	}
}

func GetInterlayerEnergy() float64 {
	buf := cuda.Buffer(1, Edens_iec.Mesh().Size())
	defer cuda.Recycle(buf)

	cuda.Zero(buf)
	AddInterlayerEnergyDensity(buf)
	return cellVolume() * float64(cuda.Sum(buf))
}
*/
func GetInterlayerEnergy() float64 {
	return -0.5 * cellVolume() * dot(&M_full, &B_iec)
}

