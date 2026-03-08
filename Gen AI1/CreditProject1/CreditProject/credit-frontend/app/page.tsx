"use client";
import { useState } from "react";

export default function Home() {

  const [age, setAge] = useState("");
  const [income, setIncome] = useState("");
  const [result, setResult] = useState("");

  const predict = async () => {

    const res = await fetch("http://127.0.0.1:8000/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        age: Number(age),
        income: Number(income)
      })
    });

    const data = await res.json();
    setResult(data.prediction);
  };

  return (
    <div style={{padding:40}}>

      <h1>Credit AI Prediction</h1>

      <input
        placeholder="Age"
        value={age}
        onChange={(e)=>setAge(e.target.value)}
      />

      <br/><br/>

      <input
        placeholder="Income"
        value={income}
        onChange={(e)=>setIncome(e.target.value)}
      />

      <br/><br/>

      <button onClick={predict}>
        Predict
      </button>

      <h2>Result: {result}</h2>

    </div>
  );
}
