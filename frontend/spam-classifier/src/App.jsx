import { useState } from 'react'
import './App.css'
import api from './api.jsx'



function App() {
  const [spamValue, setSpamValue] = useState('')
  const [score, setScore] = useState(null)
  const [email, setemail] = useState('')
  const radius = 70
  const circumference = 2*Math.PI * radius
  const offset = circumference * (1-score / 100)
 

  async function handleSubmit(){
    try {
      const response = await api.post('/predict', {email: email})

      console.log(response.data)
      setSpamValue(response.data.prediction)
      setScore(response.data.confidence)

  
    } catch (error) {
      console.log("STATUS: ", error.response?.status)
      console.log("ERROR details: ", error.response?.data)
      console.log("ERROR: ", error)
      console.log("Message: ", error.message)
      console.log("Code: ", error.code)
    }

  }

  return (
    <main className="app-container">
      
      <header className="app-header">
        <h1>Spam Email Classification</h1>
      </header>

      <section className='input-card'>
        <textarea placeholder='Paste Email Content...'
        value = {email}
        onChange={(e) => {
          setemail(e.target.value)
          setScore('')
          setSpamValue('')
        }
        }

        />
        <button onClick={handleSubmit}>
          Check Spam
        </button>
      </section>



      
      
      {score && (
        <section className="result-card">

          <h2>
            Your email is: {spamValue}
          </h2>

          <div className='confidence-container'>
            <svg className='confidence-ring' height={200} width={200} viewBox='0 0 200 200'>
            <circle className='progress-ring-background' 
            r={70} 
            cx={100} 
            cy={100} 
            fill='none' 
            stroke='darkgrey' 
            strokeWidth={15}
            />
            
            <circle className='progress-ring-circle' 
            r={70} 
            cx={100} 
            cy={100} 
            fill='none' 
            stroke='green' 
            strokeWidth={15}
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            style={{'--circumference': circumference}}
            />

            <text
            x={100}
            y={100}
            textAnchor='middle'
            fontSize={30}
            fill='currentColor'
            >
              {score}%
            </text>
            </svg>

          </div>
          Confidence

        </section>
)}
    </main>
  )

}



export default App;